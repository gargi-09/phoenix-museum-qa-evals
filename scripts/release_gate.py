"""
release_gate.py -- The CI release gate. No key, no network, no model call.

    python scripts/release_gate.py              # exit 0 = release OK, 1 = blocked

Runs the harness in replay mode (recorded candidates, recorded answers, the 8
deterministic evaluators computed live) over each fixed answer set in
samples/release_gate.json, and BLOCKS the release if:

  1. any seeded false pass gets a PASS verdict or is accepted by the harness;
  2. the known-good pass rate drops below the approved threshold;
  3. the identity regression stops being caught: the sample sent WITHOUT its
     system prompt must fail prompt_integrity, and its control (the same kind
     of question, sent WITH it) must pass it.
"""
import argparse
import asyncio
import json
import os
import sys

os.environ["LLM_PROVIDER"] = "disabled"          # before any pipeline import
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from harness.loop import run_all  # noqa: E402
from harness.replay import ReplayAnswerer, ReplayRetriever, load_samples  # noqa: E402

CONFIG_PATH = os.path.join(REPO_ROOT, "samples", "release_gate.json")


def check_set(cfg, samples):
    """[(ok, message)] for one answer set. `samples` is passed in, not read
    here, so tests can hand it a deliberately broken set."""
    by_id = {s["id"]: s for s in samples}
    missing = [i for i in cfg["seeded_false_passes"] + cfg["known_good"]
               + list(cfg["identity_regression"].values()) if i not in by_id]
    if missing:
        return [(False, f"samples missing from {cfg['path']}: {missing}")]

    wanted = sorted(set(cfg["seeded_false_passes"] + cfg["known_good"]
                        + list(cfg["identity_regression"].values())))
    states = asyncio.run(run_all([by_id[i] for i in wanted], ReplayRetriever(), ReplayAnswerer()))
    st = {s.sample_id: s for s in states}

    def verdict(i):
        return st[i].attempts[-1].verdict if st[i].attempts else None

    def evaluation(i, name):
        return st[i].attempts[-1].evaluations[name]["label"] if st[i].attempts else None

    checks = []
    approved = [i for i in cfg["seeded_false_passes"]
                if verdict(i) == "PASS" or st[i].outcome.value == "accepted"]
    n = len(cfg["seeded_false_passes"])
    checks.append((not approved, f"seeded false passes approved: {len(approved)}/{n}"
                   + (f" {approved}" if approved else f" (all {n} blocked)")))

    passed = [i for i in cfg["known_good"] if verdict(i) == "PASS"]
    need, total = cfg["min_known_good_pass"], len(cfg["known_good"])
    failing = sorted(set(cfg["known_good"]) - set(passed))
    checks.append((len(passed) >= need,
                   f"known-good pass rate: {len(passed)}/{total} (threshold {need}/{total})"
                   + (f"; failing: {failing}" if failing else "")))

    ir = cfg["identity_regression"]
    dropped, control = evaluation(ir["dropped"], "prompt_integrity"), evaluation(ir["control"], "prompt_integrity")
    checks.append((dropped == "FAIL" and control == "PASS",
                   f"identity regression: {ir['dropped']} (system prompt not sent) prompt_integrity="
                   f"{dropped}, control {ir['control']} prompt_integrity={control}"))
    return checks


def run_gate(config=None, load=load_samples):
    config = config or json.load(open(CONFIG_PATH, encoding="utf-8"))
    report = []
    for cfg in config["sets"]:
        samples = load(os.path.join(REPO_ROOT, cfg["path"]))
        report.append((cfg["name"], cfg["label"], check_set(cfg, samples)))
    ok = all(c for _, _, checks in report for c, _ in checks)
    return ok, report


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()
    ok, report = run_gate()
    for name, label, checks in report:
        print(f"[gate] {name}: {label}")
        for passed, msg in checks:
            print(f"  {'ok  ' if passed else 'FAIL'} {msg}")
    print(f"[gate] {'RELEASE OK' if ok else 'RELEASE BLOCKED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
