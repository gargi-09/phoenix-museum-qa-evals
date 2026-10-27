"""The harness loop: replay outcomes on the recorded set, and the retry path
with SCRIPTED answers (replay has one recorded attempt per sample, so retry
is only exercised here). No model is called anywhere in this file.
"""
import asyncio
import json
import os

import pytest

from conftest import RECORDED

from harness import AbstainReason, AnswerAttempt, AttemptStatus, HarnessConfig, Outcome, run_harness
from harness.loop import run_all
from harness.replay import ReplayAnswerer, ReplayRetriever, load_samples

pytestmark = pytest.mark.skipif(not os.path.exists(RECORDED), reason="needs recorded samples")

NO_WAIT = HarnessConfig(max_attempts=3, step_timeout_s=5.0, backoff_base_s=0.25, backoff_max_s=2.0)


def samples():
    return {s["id"]: s for s in load_samples(RECORDED)}


def run(coro):
    return asyncio.run(coro)


class Sleeps:
    """Stands in for asyncio.sleep: records each backoff, waits for none."""
    def __init__(self):
        self.calls = []

    async def __call__(self, seconds):
        self.calls.append(seconds)


class Scripted:
    """An Answerer with a fixed list of replies. A reply is a raw string,
    an Exception to raise, or ("sleep", seconds) to hang past the timeout."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.feedback = []

    def has_attempt(self, sample, attempt):
        return attempt <= len(self.replies)

    async def answer(self, sample, attempt, feedback):
        self.feedback.append(list(feedback))
        reply = self.replies[attempt - 1]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, tuple) and reply[0] == "sleep":
            await asyncio.sleep(reply[1])
            return AnswerAttempt("{}")
        return AnswerAttempt(reply)


def good(sample):
    return sample["payload"]["llm"]["raw_content"]


def stripped(sample):
    parsed = json.loads(good(sample))
    parsed["record_ids_used"] = []
    return json.dumps(parsed)


# --- replay on the recorded set -------------------------------------------

EXPECTED = {
    **{q: (Outcome.ACCEPTED, None) for q in ("q01", "q02", "q03", "q04", "q05", "q07",
                                             "q08", "q09", "q10")},
    "q06": (Outcome.ABSTAINED, AbstainReason.NO_FURTHER_RECORDED_ATTEMPT),  # year-check false positive
    "q11": (Outcome.ABSTAINED, AbstainReason.RETRIEVAL_GATE),
    **{s: (Outcome.ABSTAINED, AbstainReason.NO_FURTHER_RECORDED_ATTEMPT) for s in ("s01", "s02", "s03")},
}


def test_replay_outcomes_on_the_recorded_set():
    states = run(run_all(list(samples().values()), ReplayRetriever(), ReplayAnswerer()))
    got = {s.sample_id: (s.outcome, s.abstain_reason) for s in states}
    assert got == EXPECTED
    assert not any(s.retried for s in states)
    assert all(s.retrieval.source == "replayed" for s in states)


def test_seeded_false_passes_are_never_accepted():
    states = run(run_all([samples()[i] for i in ("s01", "s02", "s03")],
                         ReplayRetriever(), ReplayAnswerer()))
    assert all(s.outcome is Outcome.ABSTAINED for s in states)
    assert all(s.attempts[0].verdict == "FAIL" for s in states)


def test_parallel_evaluators_match_the_recorded_python_verdicts():
    """The harness's concurrent run gives the same 8 verdicts + roll-up as
    the sequential rescore.evaluate() that recorded them."""
    for sid, s in samples().items():
        if s["payload"]["result"].get("abstained"):
            continue
        state = run(run_harness(s, ReplayRetriever(), ReplayAnswerer()))
        ev = state.attempts[0].evaluations
        assert set(ev) == {k for k in s["python_verdicts"] if k != "llm_judge_surrogate"}, sid
        for k, v in ev.items():
            assert v["label"] == s["python_verdicts"][k]["label"], (sid, k)


def test_replay_never_calls_a_model(monkeypatch):
    import api_client

    def boom(*a, **kw):
        raise AssertionError("the harness called the model client")

    monkeypatch.setattr(api_client, "call_haiku", boom)
    run(run_all(list(samples().values()), ReplayRetriever(), ReplayAnswerer()))


# --- retry path, scripted ---------------------------------------------------

def test_retry_then_accept():
    s, sleeps = samples()["q01"], Sleeps()
    answerer = Scripted([stripped(s), good(s)])
    state = run(run_harness(s, ReplayRetriever(), answerer, NO_WAIT, sleep=sleeps))
    assert state.outcome is Outcome.ACCEPTED and state.accepted_attempt == 2 and state.retried
    assert [a.status for a in state.attempts] == [AttemptStatus.VERIFIED_FAIL,
                                                 AttemptStatus.VERIFIED_PASS]
    assert "zero_source_false_pass" in state.attempts[0].failed_evaluators
    assert sleeps.calls == [0.25]
    # The second attempt is told why the first failed.
    assert answerer.feedback[0] == [] and any("zero_source_false_pass" in f for f in answerer.feedback[1])


def test_retry_then_abstain_when_the_budget_runs_out():
    s, sleeps = samples()["q01"], Sleeps()
    state = run(run_harness(s, ReplayRetriever(), Scripted([stripped(s)] * 5), NO_WAIT, sleep=sleeps))
    assert state.outcome is Outcome.ABSTAINED
    assert state.abstain_reason is AbstainReason.RETRY_BUDGET_EXHAUSTED
    assert len(state.attempts) == 3
    assert sleeps.calls == [0.25, 0.5]                 # bounded, exponential backoff


def test_backoff_is_capped():
    assert HarnessConfig(backoff_base_s=1.0, backoff_max_s=2.0).backoff_for(6) == 2.0


def test_answer_timeout_is_an_attempt_then_retried():
    s = samples()["q01"]
    config = HarnessConfig(max_attempts=2, step_timeout_s=0.05, backoff_base_s=0.0)
    state = run(run_harness(s, ReplayRetriever(), Scripted([("sleep", 1.0), good(s)]), config,
                            sleep=Sleeps()))
    assert [a.status for a in state.attempts] == [AttemptStatus.TIMEOUT, AttemptStatus.VERIFIED_PASS]
    assert state.outcome is Outcome.ACCEPTED


def test_timeouts_until_the_budget_runs_out_abstain():
    s = samples()["q01"]
    config = HarnessConfig(max_attempts=2, step_timeout_s=0.05, backoff_base_s=0.0)
    state = run(run_harness(s, ReplayRetriever(), Scripted([("sleep", 1.0)] * 2), config,
                            sleep=Sleeps()))
    assert state.outcome is Outcome.ABSTAINED
    assert state.abstain_reason is AbstainReason.RETRY_BUDGET_EXHAUSTED
    assert all(a.status is AttemptStatus.TIMEOUT for a in state.attempts)


@pytest.mark.parametrize("malformed", [
    "The work was made in 1935 and is well documented.",      # prose, no JSON
    '{"answer": "It was made in 19',                          # truncated JSON
    "",                                                       # empty
])
def test_malformed_output_is_never_accepted(malformed):
    s = samples()["q01"]
    state = run(run_harness(s, ReplayRetriever(), Scripted([malformed]), NO_WAIT, sleep=Sleeps()))
    assert state.outcome is Outcome.ABSTAINED
    assert state.abstain_reason is AbstainReason.NO_FURTHER_RECORDED_ATTEMPT
    assert state.attempts[0].verdict == "FAIL" and state.attempts[0].failed_evaluators


def test_malformed_then_valid_is_accepted():
    s = samples()["q01"]
    state = run(run_harness(s, ReplayRetriever(), Scripted(['{"answer": "It was', good(s)]),
                            NO_WAIT, sleep=Sleeps()))
    assert state.outcome is Outcome.ACCEPTED and state.accepted_attempt == 2


def test_answerer_error_is_recorded_and_retried():
    s = samples()["q01"]
    state = run(run_harness(s, ReplayRetriever(), Scripted([RuntimeError("upstream 503"), good(s)]),
                            NO_WAIT, sleep=Sleeps()))
    assert state.attempts[0].status is AttemptStatus.ERROR and "upstream 503" in state.attempts[0].detail
    assert state.outcome is Outcome.ACCEPTED


def test_retrieval_failure_abstains():
    class Broken:
        async def retrieve(self, sample):
            raise ConnectionError("index unavailable")

    state = run(run_harness(samples()["q01"], Broken(), Scripted(["{}"]), NO_WAIT, sleep=Sleeps()))
    assert state.outcome is Outcome.ABSTAINED and state.abstain_reason is AbstainReason.RETRIEVAL_FAILED
    assert state.attempts == []


def test_trace_records_every_step_in_order():
    s = samples()["q01"]
    state = run(run_harness(s, ReplayRetriever(), Scripted([stripped(s), good(s)]), NO_WAIT,
                            sleep=Sleeps()))
    steps = [e.step.value for e in state.trace]
    assert steps == ["retrieve", "answer", "verify", "backoff", "answer", "verify", "decide"]
    assert json.dumps(state.to_dict(), default=str)    # serialisable for the metrics file
