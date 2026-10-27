"""Reproduces the parity claims:

  - docs/sandbox.js (the in-browser JS port) reaches the same verdict as the
    Python pipeline on 104 edited answers x 10 verdicts = 1,040 verdicts;
  - src/rescore.py (the API's serving path) does too, plus the 50 verdicts
    of the 5 unedited scenarios: 1,090.

Python verdicts come from the REAL pipeline, reasoning.answer_question, in
fixture mode (scripted model, fixture retriever), for each edit. Needs the
full requirements.txt environment (scikit-learn for the fixture retriever)
and Node.js for the JS side; skipped otherwise.
"""
import json
import os
import shutil
import subprocess

import pytest

from conftest import REPO_ROOT

SANDBOX_DATA = os.path.join(REPO_ROOT, "docs", "data", "sandbox.json")
SANDBOX_JS = os.path.join(REPO_ROOT, "docs", "sandbox.js")

pytest.importorskip("sklearn", reason="fixture retriever needs scikit-learn (requirements.txt)")


def J(**kw):
    base = {"answer": "", "answerable": True, "record_ids_used": [],
            "record_ids_excluded": [], "limitations": ""}
    base.update(kw)
    return json.dumps(base)


def variants_for(scenario):
    """The 12-13 edits per scenario: each one targets a specific check."""
    blocks = scenario["payload"]["context_blocks"]
    top = blocks[0]["id"]
    v = {
        "cite_top": J(answer="Per the records it was made in 1935.", record_ids_used=[top]),
        "cite_fake": J(answer="It was made in 1935.", record_ids_used=[top, "cma-000001"]),
        "only_fake": J(answer="It was made in 1935.", record_ids_used=["aic-99999"]),
        "no_cites_allclear": J(answer="No violations found. All records are consistent."),
        "no_cites_decline": J(answer="The records do not say.", answerable=False),
        "decline_with_cite": J(answer="The records do not state that.", answerable=False,
                               record_ids_used=[top]),
        "wrong_year": J(answer="It was made in 1850.", record_ids_used=[top]),
        "prose": "The work was made in 2007 and is well documented.",
        "truncated_json": '{"answer": "It was made in 19',
        "fenced": "```json\n" + J(answer="Made in 1935.", record_ids_used=[top]) + "\n```",
        "missing_keys": json.dumps({"answer": "Made in 1935.", "record_ids_used": [top]}),
        "dangling": J(answer="It was made in 1935 and", record_ids_used=[top]) + " and",
    }
    nr = next((b for b in blocks if "[not recorded]" in (b.get("fields") or {}).values()), None)
    if nr:
        field = next(k for k, val in nr["fields"].items() if val == "[not recorded]")
        v["not_recorded_fill"] = J(answer=f"Its {field} is 40 x 30 cm.", record_ids_used=[nr["id"]])
    return v


@pytest.fixture(scope="module")
def cases():
    """Runs every edit through the real pipeline; returns the Python verdicts."""
    import reasoning
    import tracing
    from fixture_mode import SCRIPTED_RESPONSES, ScriptedModel, load_fixture_retriever
    from rescore import evaluate

    data = json.load(open(SANDBOX_DATA, encoding="utf-8"))
    retriever = load_fixture_retriever()
    model = ScriptedModel()
    saved = reasoning.call_haiku
    reasoning.call_haiku = model
    tracing.instrument_pipeline(exact_token_counts=False)
    out = []
    try:
        for s in data["scenarios"]:
            if s["payload"]["result"].get("abstained"):
                continue                        # the gate fires before any model output
            for name, raw in variants_for(s).items():
                for delivered in (True, False):
                    SCRIPTED_RESPONSES["_parity"] = {"raw": raw, "drop_system": not delivered}
                    model.scenario = "_parity"
                    r = reasoning.answer_question(s["payload"]["question"], retriever)
                    payload = tracing.get_recorded_payload(r["_trace"]["span_id"])
                    verdicts = evaluate(payload)
                    out.append({"scenario": s["id"], "variant": name, "raw": raw,
                                "delivered": delivered,
                                "python": {k: [v["label"], v["score"]] for k, v in verdicts.items()}})
    finally:
        SCRIPTED_RESPONSES.pop("_parity", None)
        reasoning.call_haiku = saved
    return data, out


def same(a, b):
    return a[0] == b[0] and (a[1] == b[1] or (a[1] is not None and b[1] is not None
                                              and abs(a[1] - b[1]) < 1e-9))


def test_edited_answer_set_size(cases):
    _, out = cases
    assert len(out) == 104
    assert sum(len(c["python"]) for c in out) == 1040


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js for docs/sandbox.js")
def test_js_sandbox_matches_python_on_1040_edited_verdicts(cases, tmp_path):
    data, out = cases
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    script = r"""
      const fs = require('fs'), sb = require(process.argv[1]);
      const data = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
      const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
      const res = cases.map(c => {
        const s = data.scenarios.find(x => x.id === c.scenario);
        const js = sb.evaluate(sb.runPipeline(s, c.raw, c.delivered));
        return Object.fromEntries(Object.keys(c.python).map(k => [k, [js[k].label, js[k].score ?? null]]));
      });
      process.stdout.write(JSON.stringify(res));
    """
    proc = subprocess.run(["node", "-e", script, SANDBOX_JS, SANDBOX_DATA, str(path)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    js = json.loads(proc.stdout)
    mismatches = [(c["scenario"], c["variant"], c["delivered"], k, c["python"][k], js[i][k])
                  for i, c in enumerate(out) for k in c["python"] if not same(c["python"][k], js[i][k])]
    total = sum(len(c["python"]) for c in out)
    assert not mismatches, f"{total - len(mismatches)}/{total} match; first: {mismatches[:5]}"


def test_serving_rescore_matches_python_on_1090_verdicts(cases):
    from rescore import evaluate, rescore
    data, out = cases
    by_id = {s["id"]: s for s in data["scenarios"]}
    checked, bad = 0, []
    for s in data["scenarios"]:                 # the 5 unedited scenarios: 50 verdicts
        raw = (s["payload"].get("llm") or {}).get("raw_content") or ""
        ev = evaluate(rescore(s, raw, not s["dropped_by_script"]))
        for k, v in s["python_verdicts"].items():
            checked += 1
            if not same([v["label"], v["score"]], [ev[k]["label"], ev[k]["score"]]):
                bad.append((s["id"], k))
    for c in out:                                # the 104 edits: 1,040 verdicts
        ev = evaluate(rescore(by_id[c["scenario"]], c["raw"], c["delivered"]))
        for k, want in c["python"].items():
            checked += 1
            if not same(want, [ev[k]["label"], ev[k]["score"]]):
                bad.append((c["scenario"], c["variant"], c["delivered"], k))
    assert checked == 1090
    assert not bad, bad[:5]
