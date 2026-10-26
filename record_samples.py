"""
record_samples.py -- ONE-TIME, LOCAL-ONLY. Records real model outputs for the
live demo's fixed sample set (samples/samples.json).

    python record_samples.py            # prints the plan and the call count
    python record_samples.py --yes      # makes the paid calls and writes the files

Needs, on YOUR machine:
  - ANTHROPIC_API_KEY in .env (never committed, never deployed)
  - the public corpus: python src/main.py fetch && python src/main.py ingest
  - sentence-transformers (in requirements.txt) for the real hybrid retriever

Writes two identical files, both labelled "recorded model outputs":
  samples/recorded_outputs.json      -- what the Cloud Run API serves
  docs/data/recorded_samples.json    -- the dashboard's offline fallback

For every 'recorded' sample: one real call through the real pipeline
(reasoning.answer_question, hybrid retrieval, real prompt), captured with the
same root-span payload tracing.py writes to Phoenix -- answer, retrieved
candidates, context blocks, exact prompt-token audit (count_tokens), raw
output, usage -- plus every Python evaluator's verdict on it.

For every 'seeded' sample, exactly what samples.json declares and nothing else:
  transform                          the parent's REAL answer with one labelled
                                     change, re-scored by src/rescore.py
  real_call_without_system_prompt    a REAL call with the instruction block
                                     left out; its output and billing are real

This script never runs in the deployed service: it refuses when
LLM_PROVIDER=disabled or when Cloud Run's K_SERVICE is set, and the Docker
image does not contain it.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

SAMPLES_PATH = os.path.join(REPO_ROOT, "samples", "samples.json")
OUT_PATHS = [os.path.join(REPO_ROOT, "samples", "recorded_outputs.json"),
             os.path.join(REPO_ROOT, "docs", "data", "recorded_samples.json")]
NORMALIZED_PATH = os.path.join(REPO_ROOT, "data", "normalized.jsonl")
SOURCES_PATH = os.path.join(REPO_ROOT, "data", "SOURCES.json")


def refuse(msg):
    print(f"[record] REFUSING: {msg}")
    sys.exit(2)


def guard():
    if os.getenv("K_SERVICE") or os.getenv("LLM_PROVIDER", "").lower() == "disabled":
        refuse("this is the serving environment. Recording is local-only.")
    import api_client
    if api_client.PROVIDER != "anthropic" or not api_client.check_credentials(quiet=True):
        refuse("needs ANTHROPIC_API_KEY in .env (Anthropic provider). Nothing was called.")
    if not os.path.exists(NORMALIZED_PATH):
        refuse(f"{NORMALIZED_PATH} is missing. Run: python src/main.py fetch && "
               f"python src/main.py ingest")
    return api_client


def audit_inputs(payload):
    """with/without-instructions estimates, from the recorded audit:
    billed = with + delta, instructions = with - without."""
    audit = payload.get("prompt_audit") or {}
    billed = payload.get("prompt_tokens")
    if billed is None or "prompt_tokens_delta" not in audit:
        return None
    with_i = billed - audit["prompt_tokens_delta"]
    return {"expected_with_instructions": with_i,
            "expected_without_instructions": with_i - audit["instruction_tokens_expected"],
            "instruction_tokens": audit["instruction_tokens_expected"]}


def transform_answer(raw, spec):
    """The only edits a seeded sample is allowed: declared in samples.json."""
    import reasoning
    parsed, err = reasoning.parse_haiku_response(raw)
    if parsed is None:
        raise RuntimeError(f"parent answer did not parse ({err}); cannot apply {spec['transform']}")
    parsed = dict(parsed)
    if spec["transform"] == "strip_citations":
        parsed["record_ids_used"] = []
    elif spec["transform"] == "add_citation":
        parsed["record_ids_used"] = list(parsed.get("record_ids_used") or []) + [spec["record_id"]]
    else:
        raise ValueError(f"unknown transform {spec['transform']!r}")
    return json.dumps(parsed, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser(description="Record real model outputs for the demo samples.")
    p.add_argument("--yes", action="store_true", help="make the paid calls")
    args = p.parse_args()

    spec = json.load(open(SAMPLES_PATH, encoding="utf-8"))["samples"]
    recorded = [s for s in spec if s["kind"] == "recorded"]
    real_seeds = [s for s in spec if s["kind"] == "seeded"
                  and s["seed"]["type"] == "real_call_without_system_prompt"]
    api_client = guard()
    print(f"[record] {len(recorded)} recorded + {len(real_seeds)} real seeded call(s) "
          f"on {api_client.active_model()} (questions the gate abstains on make no call). "
          f"Plus free count_tokens calls for the prompt audit.")
    if not args.yes:
        print("[record] Dry run. Re-run with --yes to make the calls.")
        return 0

    import reasoning
    import tracing
    from rescore import evaluate, rescore
    from retrieval import HybridRetriever, load_records

    retriever = HybridRetriever(load_records(NORMALIZED_PATH))

    # Installed BEFORE instrumentation so the LLM span wraps it. Only s02's
    # call drops the instructions, and only because samples.json says so.
    real_call = reasoning.call_haiku
    drop = {"on": False}

    def call(messages, **kw):
        if drop["on"]:
            kw["system"] = None
        return real_call(messages, **kw)

    reasoning.call_haiku = call
    tracing.instrument_pipeline(exact_token_counts=True)
    tracker = api_client.get_tracker()

    def record(sample, question, drop_system=False):
        drop["on"] = drop_system
        n_calls = len(tracker.log)
        try:
            result = reasoning.answer_question(question, retriever)
        finally:
            drop["on"] = False
        payload = tracing.get_recorded_payload(result["_trace"]["span_id"])
        if result.get("error") and (payload.get("llm") or {}).get("error_type"):
            raise RuntimeError(f"{sample['id']}: model call failed: {result['error']}")
        _, _, best_sim, _ = retriever.retrieve(question, top_k=10)
        calls = tracker.log[n_calls:]
        return {
            "payload": payload,
            "audit_inputs": audit_inputs(payload),
            "best_similarity": best_sim,
            "max_tokens": reasoning.ANSWER_MAX_TOKENS,
            "dropped_by_script": drop_system,
            "recorded": {"model": api_client.active_model(), "provider": "anthropic",
                         "recorded_at": datetime.now(timezone.utc).isoformat(),
                         "model_calls": len(calls),
                         "tokens": sum(c["tokens"] for c in calls)},
        }

    out, by_id = [], {}
    for s in spec:
        if s["kind"] == "recorded":
            print(f"[record] {s['id']}: {s['question']}")
            entry = record(s, s["question"])
        else:
            seed = s["seed"]
            parent = by_id[seed["from"]]
            if seed["type"] == "real_call_without_system_prompt":
                print(f"[record] {s['id']}: real call WITHOUT the system prompt ({seed['from']})")
                entry = record(s, parent["question"], drop_system=True)
            else:
                print(f"[record] {s['id']}: {seed['transform']} on {seed['from']}'s recorded answer")
                raw = transform_answer(parent["payload"]["llm"]["raw_content"], seed)
                entry = {k: parent[k] for k in ("audit_inputs", "best_similarity", "max_tokens")}
                entry["payload"] = rescore(parent, raw, system_prompt_delivered=True)
                entry["dropped_by_script"] = False
                entry["recorded"] = {"model": parent["recorded"]["model"], "provider": "anthropic",
                                     "recorded_at": parent["recorded"]["recorded_at"],
                                     "model_calls": 0, "tokens": 0,
                                     "derived_from": seed["from"]}
        entry.update({"id": s["id"], "kind": s["kind"], "expect": s["expect"],
                      "question": entry["payload"]["question"], "seed": s.get("seed")})
        entry["python_verdicts"] = evaluate(entry["payload"])
        by_id[s["id"]] = entry
        out.append(entry)
        v = entry["python_verdicts"]
        print(f"         verdict={v['deterministic_verdict']['label']:<15} "
              f"judge={v['llm_judge_surrogate']['label']}")

    doc = {
        "label": "recorded model outputs",
        "notice": ("Every 'recorded' answer is a real model output captured by "
                   "record_samples.py; 'seeded' samples are labelled transforms or a "
                   "labelled real call without the system prompt. The live API serves "
                   "these and computes the evaluators live; it never calls a model."),
        "generated_by": "record_samples.py",
        "model": api_client.active_model(),
        "corpus": json.load(open(SOURCES_PATH, encoding="utf-8")) if os.path.exists(SOURCES_PATH) else None,
        "total_model_calls": sum(e["recorded"]["model_calls"] for e in out),
        "total_tokens": sum(e["recorded"]["tokens"] for e in out),
        "scenarios": out,
    }
    for path in OUT_PATHS:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print(f"[record] wrote {os.path.relpath(path, REPO_ROOT)}")
    print(f"[record] {doc['total_model_calls']} model call(s), {doc['total_tokens']} tokens.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
