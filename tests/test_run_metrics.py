"""scripts/run_metrics.py: every number in the metrics file is either measured
by the run or copied from the recording, and labelled as which. No model call.
"""
import json
import os
import sys

import pytest

from conftest import RECORDED, REPO_ROOT

sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import run_metrics  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(RECORDED), reason="needs recorded samples")

COMMITTED = os.path.join(REPO_ROOT, "docs", "data", "metrics.json")


@pytest.fixture(scope="module")
def doc():
    return run_metrics.build(repeats=2)


def test_outcomes_match_the_harness_on_the_recorded_set(doc):
    s = doc["summary"]
    assert s["samples"] == 14
    assert s["harness_outcomes"] == {"accepted": 9, "abstained": 5, "retried": 0}
    assert s["abstain_reasons"] == {"no_further_recorded_attempt": 4, "retrieval_gate": 1}
    by_id = {q["id"]: q for q in doc["questions"]}
    assert all(by_id[i]["harness"]["outcome"] == "abstained" for i in ("s01", "s02", "s03", "q06"))


def test_evaluator_counts_add_up(doc):
    ev = doc["summary"]["evaluators"]
    assert len(ev) == 9                                    # the 8 + deterministic_verdict
    generated = sum(1 for q in doc["questions"] if q["evaluators"])
    assert all(sum(c.values()) == generated for c in ev.values())
    assert ev["deterministic_verdict"]["PASS"] == 9


def test_tokens_are_the_recorded_ones(doc):
    rec = json.load(open(RECORDED, encoding="utf-8"))
    t = doc["summary"]["recorded_tokens_per_answer"]
    assert t["sum_total_tokens"] == rec["total_tokens"] == 54954
    assert t["answers_with_a_model_call"] == rec["total_model_calls"] == 11
    assert (t["prompt_tokens"]["min"], t["prompt_tokens"]["max"]) == (3694, 4900)
    assert "Recorded, not measured" in doc["notes"]["tokens"]


def test_latency_is_labelled_as_replay_only(doc):
    lat = doc["summary"]["latency_ms"]
    assert lat["all_runs"]["n"] == 14 * 2
    assert lat["all_runs"]["p50_ms"] > 0 and lat["all_runs"]["p95_ms"] >= lat["all_runs"]["p50_ms"]
    assert "does NOT include a model call" in doc["notes"]["latency"].replace("Does", "does")
    assert set(lat["per_step"]) == {"retrieve", "answer", "verify", "decide"}


def test_committed_metrics_file_matches_a_fresh_run(doc):
    """Latency varies by machine; everything else in the committed file must
    be what a fresh run produces today."""
    if not os.path.exists(COMMITTED):
        pytest.skip("no committed metrics file")
    old = json.load(open(COMMITTED, encoding="utf-8"))
    assert old["label"] == "measured metrics (replay mode)"
    for key in ("harness_outcomes", "abstain_reasons", "evaluators",
                "recorded_tokens_per_answer", "recorded_llm_judge"):
        assert old["summary"][key] == doc["summary"][key], key
    strip = lambda qs: [{k: v for k, v in q.items() if k != "latency_ms"} for q in qs]  # noqa: E731
    assert strip(old["questions"]) == strip(doc["questions"])
