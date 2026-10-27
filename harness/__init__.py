"""A small harness around the existing pipeline pieces:

    retrieve -> answer -> verify (the 8 deterministic evaluators) ->
    accept | retry (bounded, with backoff) | abstain -> trace

It does not change reasoning.answer_question, which still makes one model
call per answer. On the public service it runs in REPLAY mode
(harness.replay): recorded candidates, recorded answers, live evaluators.
"""
from harness.loop import run_all, run_harness, verify
from harness.types import (AbstainReason, AnswerAttempt, Attempt, AttemptStatus, HarnessConfig,
                           HarnessState, NoFurtherAttempt, Outcome, Retrieval)

__all__ = ["run_all", "run_harness", "verify", "AbstainReason", "AnswerAttempt", "Attempt",
           "AttemptStatus", "HarnessConfig", "HarnessState", "NoFurtherAttempt", "Outcome",
           "Retrieval"]
