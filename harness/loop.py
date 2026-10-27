"""The harness loop.

The gate is the pipeline's own deterministic evaluators: an attempt is
accepted only when deterministic_verdict (the roll-up of all 8) is PASS. No
LLM judge takes part in the decision.

The 8 evaluators run concurrently (asyncio.gather over worker threads). They
take about 0.3 ms in total, so this is about structure, not speed: each one
is isolated, and one that raises is reported without hiding the others.

Nothing here calls a model. The Answerer decides where answers come from;
on the public service that is harness.replay.ReplayAnswerer.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from harness.types import (AbstainReason, Answerer, Attempt, AttemptStatus, HarnessConfig,
                           HarnessState, NoFurtherAttempt, Outcome, Retriever, Step, TraceEvent)

Sleep = Callable[[float], Awaitable[None]]


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 3)


async def verify(sample: dict[str, Any], raw_output: str,
                 system_prompt_delivered: bool) -> tuple[str, dict[str, dict[str, Any]]]:
    """(deterministic_verdict label, {evaluator: {label, score, explanation}})."""
    from phoenix_evals import DETERMINISTIC_EVALUATORS, deterministic_verdict, run_evaluators
    from rescore import rescore

    payload = await asyncio.to_thread(rescore, sample, raw_output, system_prompt_delivered)
    parts = await asyncio.gather(*(
        asyncio.to_thread(run_evaluators, payload, {name: fn})
        for name, fn in DETERMINISTIC_EVALUATORS.items()))
    results = {name: r for part in parts for name, r in part.items()}
    verdict = deterministic_verdict(results)
    out = {k: v.to_dict() for k, v in results.items()}
    out["deterministic_verdict"] = verdict.to_dict()
    return verdict.label, out


async def run_harness(sample: dict[str, Any], retriever: Retriever, answerer: Answerer,
                      config: HarnessConfig = HarnessConfig(),
                      sleep: Sleep = asyncio.sleep) -> HarnessState:
    state = HarnessState(sample_id=sample.get("id", "?"),
                         question=(sample.get("payload") or {}).get("question", ""))
    t_run = time.perf_counter()

    def event(step: Step, attempt: int | None, t0: float, detail: str = "") -> None:
        state.trace.append(TraceEvent(step, attempt, _ms(t0), detail))

    def finish(outcome: Outcome, reason: AbstainReason | None = None) -> HarnessState:
        state.outcome, state.abstain_reason = outcome, reason
        t = time.perf_counter()
        event(Step.DECIDE, state.accepted_attempt, t,
              outcome.value if reason is None else f"{outcome.value}: {reason.value}")
        state.total_ms = _ms(t_run)
        return state

    # retrieve
    t0 = time.perf_counter()
    try:
        state.retrieval = await asyncio.wait_for(retriever.retrieve(sample), config.step_timeout_s)
    except (asyncio.TimeoutError, Exception) as e:  # noqa: BLE001 -- becomes an abstention
        event(Step.RETRIEVE, None, t0, f"failed: {type(e).__name__}")
        return finish(Outcome.ABSTAINED, AbstainReason.RETRIEVAL_FAILED)
    event(Step.RETRIEVE, None, t0, f"{state.retrieval.source}, "
          f"{len(state.retrieval.candidate_ids)} candidates")
    if state.retrieval.abstained:
        return finish(Outcome.ABSTAINED, AbstainReason.RETRIEVAL_GATE)

    feedback: list[str] = []
    for n in range(1, config.max_attempts + 1):
        if not answerer.has_attempt(sample, n):
            return finish(Outcome.ABSTAINED, AbstainReason.NO_FURTHER_RECORDED_ATTEMPT)
        if n > 1:
            t0 = time.perf_counter()
            wait = config.backoff_for(n)
            await sleep(wait)
            event(Step.BACKOFF, n, t0, f"{wait:.3f}s")

        # answer
        t0 = time.perf_counter()
        try:
            ans = await asyncio.wait_for(answerer.answer(sample, n, feedback), config.step_timeout_s)
        except NoFurtherAttempt:
            event(Step.ANSWER, n, t0, "no further attempt")
            return finish(Outcome.ABSTAINED, AbstainReason.NO_FURTHER_RECORDED_ATTEMPT)
        except asyncio.TimeoutError:
            event(Step.ANSWER, n, t0, "timeout")
            state.attempts.append(Attempt(n, AttemptStatus.TIMEOUT,
                                          detail=f"answer exceeded {config.step_timeout_s}s"))
            feedback = ["previous attempt timed out"]
            continue
        except Exception as e:  # noqa: BLE001 -- recorded on the attempt, then retried
            event(Step.ANSWER, n, t0, f"error: {type(e).__name__}")
            state.attempts.append(Attempt(n, AttemptStatus.ERROR, detail=f"{type(e).__name__}: {e}"))
            feedback = [f"previous attempt raised {type(e).__name__}"]
            continue
        event(Step.ANSWER, n, t0, f"{len(ans.raw_output or '')} chars")

        # verify
        t0 = time.perf_counter()
        try:
            label, evaluations = await asyncio.wait_for(
                verify(sample, ans.raw_output, ans.system_prompt_delivered), config.step_timeout_s)
        except asyncio.TimeoutError:
            event(Step.VERIFY, n, t0, "timeout")
            state.attempts.append(Attempt(n, AttemptStatus.TIMEOUT, raw_output=ans.raw_output,
                                          detail=f"verify exceeded {config.step_timeout_s}s"))
            feedback = ["previous attempt could not be verified in time"]
            continue
        failed = [k for k, v in evaluations.items()
                  if v["label"] == "FAIL" and k != "deterministic_verdict"]
        event(Step.VERIFY, n, t0, f"{label}" + (f": {', '.join(failed)}" if failed else ""))
        ok = label == "PASS"
        state.attempts.append(Attempt(
            n, AttemptStatus.VERIFIED_PASS if ok else AttemptStatus.VERIFIED_FAIL,
            raw_output=ans.raw_output, verdict=label, failed_evaluators=failed,
            evaluations=evaluations))
        if ok:
            state.accepted_attempt = n
            return finish(Outcome.ACCEPTED)
        feedback = [f"{k}: {evaluations[k]['explanation']}" for k in failed]

    return finish(Outcome.ABSTAINED, AbstainReason.RETRY_BUDGET_EXHAUSTED)


async def run_all(samples: list[dict[str, Any]], retriever: Retriever, answerer: Answerer,
                  config: HarnessConfig = HarnessConfig(),
                  sleep: Sleep = asyncio.sleep) -> list[HarnessState]:
    return list(await asyncio.gather(*(run_harness(s, retriever, answerer, config, sleep)
                                       for s in samples)))
