"""POST /gate/run: the CI release gate (scripts/release_gate.py), run live on
the recorded answer set, with optional deliberate regressions so a visitor
can watch it block a release. No model call: the gate replays recorded
answers through the harness and the 8 evaluators.

Each regression breaks a COPY of the data in memory, exactly as
tests/test_release_gate.py does; nothing on disk changes.
"""
import copy
import importlib.util
import json
import os
import time

BREAKS = {
    "none": "No change: the committed answer set.",
    "approve_seeded": "s01 gets q01's real, cited answer, so a seeded false pass now verifies.",
    "drop_known_good": "q01's real answer loses its citations. With q06's known false positive, known-good drops to 5/7, below the 6/7 threshold.",
    "hide_dropped_prompt": "s02 is billed as if its system prompt arrived, so the drop goes unseen.",
}


def _load_gate(repo_root):
    spec = importlib.util.spec_from_file_location(
        "release_gate", os.path.join(repo_root, "scripts", "release_gate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _strip(sample):
    parsed = json.loads(sample["payload"]["llm"]["raw_content"])
    parsed["record_ids_used"] = []
    sample["payload"]["llm"]["raw_content"] = json.dumps(parsed)


def _apply(name, samples):
    by_id = {s["id"]: s for s in samples}
    if name == "approve_seeded":
        by_id["s01"]["payload"]["llm"]["raw_content"] = by_id["q01"]["payload"]["llm"]["raw_content"]
    elif name == "drop_known_good":
        _strip(by_id["q01"])
    elif name == "hide_dropped_prompt":
        by_id["s02"]["dropped_by_script"] = False
    return samples


class GateRunner:
    def __init__(self, repo_root, recorded_samples):
        self.gate = _load_gate(repo_root)
        self.config = json.load(open(self.gate.CONFIG_PATH, encoding="utf-8"))
        self.cfg = next(c for c in self.config["sets"] if c["name"] == "recorded")
        self.samples = recorded_samples

    def run(self, break_name="none"):
        if break_name not in BREAKS:
            raise KeyError(break_name)
        t0 = time.perf_counter()
        samples = _apply(break_name, copy.deepcopy(self.samples))
        checks = self.gate.check_set(self.cfg, samples)
        ok = all(c for c, _ in checks)
        return {
            "break": break_name, "break_description": BREAKS[break_name],
            "set": self.cfg["name"], "label": self.cfg["label"],
            "thresholds": {"seeded_false_passes": self.cfg["seeded_false_passes"],
                           "known_good": self.cfg["known_good"],
                           "min_known_good_pass": self.cfg["min_known_good_pass"],
                           "identity_regression": self.cfg["identity_regression"]},
            "checks": [{"ok": c, "message": m} for c, m in checks],
            "release": "OK" if ok else "BLOCKED",
            "ran_in_ms": round((time.perf_counter() - t0) * 1000, 1),
            "note": ("The same gate GitHub Actions runs on every push "
                     "(scripts/release_gate.py), on the recorded answers. No model call."),
        }
