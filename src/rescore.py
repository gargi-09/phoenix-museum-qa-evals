"""
rescore.py -- Re-runs everything AFTER the model call, for a recorded sample
and a (possibly edited) model output. No retrieval, no model call.

This is the serving path's only pipeline code. It composes the pipeline's own
functions rather than re-implementing them:

    reasoning.parse_haiku_response      -> the parsed answer, or a parse error
    api_client.detect_truncation        -> truncation signals
    reasoning.run_verification_suite    -> the four deterministic checks
    reasoning.confidence_for_response   -> the computed confidence
    tracing.audit_prompt_tokens         -> the dropped-system-prompt audit
    phoenix_evals.*                     -> the 8 evaluators, verdict, judge baseline

and assembles them in the same order and shape as reasoning.answer_question()
and tracing.py's root-span payload. docs/sandbox.js is the browser port of this
same sequence (its runPipeline()).

Retrieval is replayed, not re-run: a sample carries the candidate IDs and
context blocks recorded on its trace, so an edited answer is checked against
exactly the records the pipeline retrieved for that question.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def rescore(sample, raw_content, system_prompt_delivered=True):
    """Returns an eval payload (tracing.py's root-span shape) for `raw_content`
    answering `sample`'s question. An abstained sample is returned unchanged:
    the gate fired before any model call, so there is no output to edit."""
    import reasoning
    from api_client import detect_truncation
    from tracing import audit_prompt_tokens

    base = sample["payload"]
    if (base.get("result") or {}).get("abstained"):
        return base

    inputs = sample.get("audit_inputs") or {}
    with_i = inputs.get("expected_with_instructions")
    without_i = inputs.get("expected_without_instructions")
    billed = with_i if system_prompt_delivered else without_i
    audit = (audit_prompt_tokens(billed, with_i, without_i)
             if with_i is not None and without_i is not None
             else {"status": "unmeasured", "system_prompt_dropped_suspected": False,
                   "detail": "this sample recorded no prompt-token estimates"})

    max_tokens = sample.get("max_tokens") or reasoning.ANSWER_MAX_TOKENS
    # An edited answer has no billed completion count, so use the same
    # ~4 chars/token estimate the fixture model bills by.
    usage = {"completion_tokens": max(1, len(raw_content or "") // 4)}
    is_truncated, reasons = detect_truncation(raw_content, usage, max_tokens,
                                              expect_json=True)
    llm = {"error_type": None, "is_truncated": is_truncated,
           "truncation_reasons": reasons, "prior_truncation": None,
           "truncation_recovered": False, "raw_content": raw_content}

    blocks = base.get("context_blocks") or []
    parsed, parse_error = reasoning.parse_haiku_response(raw_content)

    if parse_error and not parsed:
        result = {"question": base["question"], "abstained": False,
                  "error": f"failed_to_parse_response: {parse_error}",
                  "answer": None, "raw_content": raw_content}
    else:
        checks, passed, total = reasoning.run_verification_suite(
            parsed, blocks, haiku_result=llm)
        used = set(parsed.get("record_ids_used") or [])
        has_contradiction = any(b.get("internal_contradiction") for b in blocks
                                if b["id"] in used)
        confidence = reasoning.confidence_for_response(
            parsed, sample.get("best_similarity") or 0.0, passed, total,
            has_contradiction)
        result = {"question": base["question"], "abstained": False,
                  "answer": parsed.get("answer"),
                  "answerable": parsed.get("answerable"),
                  "record_ids_used": parsed.get("record_ids_used"),
                  "limitations": parsed.get("limitations"),
                  "confidence": confidence,
                  "verification_checks": checks,
                  "is_truncated": is_truncated}

    return {"question": base["question"], "result": result,
            "candidate_ids": base.get("candidate_ids") or [],
            "context_blocks": blocks, "prompt_audit": audit,
            "prompt_tokens": billed, "llm": llm}


def evaluate(payload):
    """All 8 deterministic evaluators, the roll-up verdict and the
    reference-free judge baseline, as {name: {label, score, explanation}}."""
    from phoenix_evals import (DETERMINISTIC_EVALUATORS, deterministic_verdict,
                               llm_judge_surrogate, run_evaluators)
    results = run_evaluators(payload, DETERMINISTIC_EVALUATORS)
    results["deterministic_verdict"] = deterministic_verdict(results)
    results["llm_judge_surrogate"] = llm_judge_surrogate(payload)
    return {k: v.to_dict() for k, v in results.items()}
