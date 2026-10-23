"""
export_sandbox_data.py -- Writes docs/data/sandbox.json, the data behind the
dashboard's in-browser Evaluator Sandbox.

    python src/export_sandbox_data.py

Runs the five demo questions through the REAL pipeline in fixture mode (same
code path as `python demo_phoenix_evals.py`, minus Phoenix), and saves for
each one:
  - the eval payload exactly as tracing.py writes it onto the root span
    (question, pipeline result, RRF candidate ids, context blocks, prompt
    audit, raw model output);
  - the prompt-token numbers the sandbox needs to re-run the audit when a
    visitor toggles "system prompt delivered";
  - every Python evaluator's verdict, so the page can check on load that its
    JavaScript port reaches the same verdicts (docs/sandbox.js).

No network, no credentials, no Phoenix server: with no tracer registered,
OpenTelemetry's no-op tracer is used and the payloads are still recorded.
"""
import json
import os
import sys

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

OUT_PATH = os.path.join(REPO_ROOT, "docs", "data", "sandbox.json")

# Sandbox ids line up with the dashboard's scenario ids where they overlap.
SANDBOX_IDS = {
    "clean": "clean",
    "zero_source_false_pass": "zero",
    "dropped_system_prompt": "dropped",
    "fabricated_citation": "fabricated",
    "out_of_corpus_decline": "abstain",
}


def main():
    import reasoning
    import tracing
    from api_client import MESSAGE_FORMAT, estimate_tokens_preflight
    from demo_phoenix_evals import DEMO_QUERIES
    from fixture_mode import ScriptedModel, load_fixture_retriever
    from phoenix_evals import (DETERMINISTIC_EVALUATORS, deterministic_verdict,
                               llm_judge_surrogate, run_evaluators)

    retriever = load_fixture_retriever()
    model = ScriptedModel()
    reasoning.call_haiku = model
    tracing.instrument_pipeline(exact_token_counts=False)

    scenarios = []
    for question, scenario, why in DEMO_QUERIES:
        model.scenario = scenario
        result = reasoning.answer_question(question, retriever)
        payload = tracing.get_recorded_payload(result["_trace"]["span_id"])
        if payload is None:
            raise RuntimeError(f"no payload recorded for {question!r}")

        evals = run_evaluators(payload, DETERMINISTIC_EVALUATORS)
        evals["deterministic_verdict"] = deterministic_verdict(evals)
        evals["llm_judge_surrogate"] = llm_judge_surrogate(payload)

        # Token numbers for the audit, so the sandbox can flip "delivered"
        # without re-running retrieval: billed = with - delta on this path.
        audit = payload.get("prompt_audit") or {}
        billed = payload.get("prompt_tokens")
        expected_with = (billed - audit["prompt_tokens_delta"]
                         if billed is not None and "prompt_tokens_delta" in audit else None)
        instruction_tokens = audit.get("instruction_tokens_expected")
        # Retrieval strength feeds confidence; an errored result does not
        # carry it, so take it from the retriever directly (same question,
        # deterministic retriever, same number).
        _, _, best_sim, _ = retriever.retrieve(question, top_k=10)
        scenarios.append({
            "id": SANDBOX_IDS.get(scenario, scenario),
            "scenario": scenario,
            "why": why,
            "payload": payload,
            "dropped_by_script": bool((fixture_spec(scenario) or {}).get("drop_system")),
            "message_format": MESSAGE_FORMAT,
            "python_verdicts": {k: v.to_dict() for k, v in evals.items()},
            "audit_inputs": {
                "expected_with_instructions": expected_with,
                "expected_without_instructions": (expected_with - instruction_tokens
                                                  if expected_with is not None else None),
                "instruction_tokens": instruction_tokens,
            },
            "best_similarity": best_sim,
            "max_tokens": reasoning.ANSWER_MAX_TOKENS,
        })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "generated_by": "src/export_sandbox_data.py",
            "source": "fixture mode: 10 public museum records, scripted model",
            "scenarios": scenarios,
        }, f, ensure_ascii=False, indent=1)
    print(f"[export] Wrote {len(scenarios)} scenarios to "
          f"{os.path.relpath(OUT_PATH, REPO_ROOT)}")
    for s in scenarios:
        v = s["python_verdicts"]
        print(f"  {s['id']:<11} verdict={v['deterministic_verdict']['label']:<15} "
              f"judge={v['llm_judge_surrogate']['label']}")


def fixture_spec(scenario):
    from fixture_mode import SCRIPTED_RESPONSES
    return SCRIPTED_RESPONSES.get(scenario)


if __name__ == "__main__":
    main()
