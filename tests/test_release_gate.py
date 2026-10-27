"""The release gate passes on the committed answer sets, and BLOCKS each
regression it exists for. Every failing case below breaks a copy of the
data on purpose; nothing on disk changes. No model is called.
"""
import copy
import json
import os
import subprocess
import sys

import pytest

from conftest import RECORDED, REPO_ROOT

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import release_gate  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(RECORDED), reason="needs recorded samples")

CONFIG = json.load(open(release_gate.CONFIG_PATH, encoding="utf-8"))
RECORDED_CFG = next(c for c in CONFIG["sets"] if c["name"] == "recorded")


def recorded():
    return copy.deepcopy(json.load(open(RECORDED, encoding="utf-8"))["scenarios"])


def by_id(samples):
    return {s["id"]: s for s in samples}


def failed(checks):
    return [msg for ok, msg in checks if not ok]


def strip_citations(sample):
    parsed = json.loads(sample["payload"]["llm"]["raw_content"])
    parsed["record_ids_used"] = []
    sample["payload"]["llm"]["raw_content"] = json.dumps(parsed)


def test_gate_passes_on_the_committed_sets():
    ok, report = release_gate.run_gate()
    assert ok, report
    assert [name for name, _, _ in report] == ["recorded", "fixture"]


def test_gate_cli_exit_code():
    proc = subprocess.run([sys.executable, os.path.join("scripts", "release_gate.py")],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RELEASE OK" in proc.stdout


def test_approved_thresholds_are_the_ones_on_record():
    assert RECORDED_CFG["known_good"] == [f"q0{i}" for i in range(1, 8)]
    assert RECORDED_CFG["min_known_good_pass"] == 6           # approved: >= 6/7
    assert RECORDED_CFG["seeded_false_passes"] == ["s01", "s02", "s03"]


def test_blocks_a_seeded_false_pass_that_passes():
    samples = recorded()
    s = by_id(samples)
    # s01 now carries q01's real, cited answer: the seeded case no longer fails.
    s["s01"]["payload"]["llm"]["raw_content"] = s["q01"]["payload"]["llm"]["raw_content"]
    msgs = failed(release_gate.check_set(RECORDED_CFG, samples))
    assert len(msgs) == 1 and "seeded false passes approved: 1/3 ['s01']" in msgs[0]


def test_blocks_a_drop_in_the_known_good_pass_rate():
    samples = recorded()
    strip_citations(by_id(samples)["q01"])                   # 5/7 < 6/7
    msgs = failed(release_gate.check_set(RECORDED_CFG, samples))
    assert len(msgs) == 1 and "known-good pass rate: 5/7" in msgs[0]


def test_six_of_seven_is_exactly_at_the_threshold():
    msgs = failed(release_gate.check_set(RECORDED_CFG, recorded()))
    assert msgs == []                                         # q06 failing, 6/7 passes


def test_blocks_the_identity_regression_going_undetected():
    samples = recorded()
    # Pretend s02 was sent WITH its instructions: its billed prompt_tokens now
    # look normal, so prompt_integrity would no longer flag the drop.
    by_id(samples)["s02"]["dropped_by_script"] = False
    msgs = failed(release_gate.check_set(RECORDED_CFG, samples))
    assert any("identity regression" in m and "prompt_integrity=PASS" in m for m in msgs)


def test_blocks_an_audit_that_flags_everything():
    samples = recorded()
    by_id(samples)["q03"]["dropped_by_script"] = True        # the control now looks dropped
    msgs = failed(release_gate.check_set(RECORDED_CFG, samples))
    assert any("identity regression" in m and "control q03 prompt_integrity=FAIL" in m for m in msgs)


def test_blocks_a_missing_sample():
    samples = [s for s in recorded() if s["id"] != "s02"]
    msgs = failed(release_gate.check_set(RECORDED_CFG, samples))
    assert msgs and "missing" in msgs[0]


def test_ci_workflow_runs_the_gate_without_secrets():
    wf = open(os.path.join(REPO_ROOT, ".github", "workflows", "release-gate.yml"),
              encoding="utf-8").read()
    assert "LLM_PROVIDER: disabled" in wf
    assert "secrets." not in wf and "ANTHROPIC" not in wf
    assert "python scripts/release_gate.py" in wf and "python -m pytest" in wf
    assert "pip install -r requirements-ci.txt" in wf
    ci = open(os.path.join(REPO_ROOT, "requirements-ci.txt"), encoding="utf-8").read().lower()
    for pkg in ("anthropic", "requests", "python-dotenv", "torch", "sentence-transformers",
                "arize-phoenix"):
        assert f"\n{pkg}=" not in ci


def test_ci_checks_the_compiled_sandbox_matches_its_typescript_source():
    wf = open(os.path.join(REPO_ROOT, ".github", "workflows", "release-gate.yml"),
              encoding="utf-8").read()
    assert "npx tsc -p tsconfig.json" in wf and "git diff --exit-code -- docs/sandbox.js" in wf
    cfg = json.load(open(os.path.join(REPO_ROOT, "tsconfig.json"), encoding="utf-8"))["compilerOptions"]
    assert cfg["strict"] is True and cfg["noImplicitAny"] is True
    pkg = json.load(open(os.path.join(REPO_ROOT, "package.json"), encoding="utf-8"))
    assert pkg["devDependencies"] == {"typescript": "5.9.3"}           # pinned, the only dependency
