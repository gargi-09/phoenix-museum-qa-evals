"""
run_metrics.py -- Runs the full fixed set through the harness and writes what
it MEASURED to docs/data/metrics.json. No key, no network, no model call.

    python scripts/run_metrics.py                 # 50 timed runs per sample
    python scripts/run_metrics.py --repeats 200 --out path.json

What each number is, and is not:
  - evaluator results: the 8 deterministic evaluators + deterministic_verdict,
    computed now, by this run, on each sample's recorded answer.
  - harness outcomes: accepted / retried / abstained, from this run (replay).
  - latency: wall-clock time of one harness run in REPLAY mode on THIS
    machine. It covers retrieval replay (recorded candidates), the
    evaluators and the harness loop. It does NOT include a model call or any
    network: there is none. The first run, which includes importing the
    pipeline modules, is reported separately and left out of p50/p95.
  - tokens per answer: RECORDED by record_samples.py when the answer was
    made, copied from samples/recorded_outputs.json. Not measured here.
"""
import argparse
import asyncio
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

os.environ["LLM_PROVIDER"] = "disabled"          # before any pipeline import
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from harness.loop import run_harness  # noqa: E402
from harness.replay import ReplayAnswerer, ReplayRetriever, load_samples  # noqa: E402

SAMPLES_PATH = os.path.join(REPO_ROOT, "samples", "recorded_outputs.json")
JUDGE_PATH = os.path.join(REPO_ROOT, "samples", "recorded_llm_judge.json")
OUT_PATH = os.path.join(REPO_ROOT, "docs", "data", "metrics.json")


def pct(values, p):
    """p-th percentile, statistics.quantiles(method='inclusive')."""
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1]


def summary(values):
    return {"n": len(values), "p50_ms": round(pct(values, 50), 3),
            "p95_ms": round(pct(values, 95), 3), "max_ms": round(max(values), 3)}


async def timed_runs(samples, repeats):
    retriever, answerer = ReplayRetriever(), ReplayAnswerer()
    t0 = time.perf_counter()
    await run_harness(samples[0], retriever, answerer)      # imports the pipeline
    first_ms = (time.perf_counter() - t0) * 1000
    per_sample, steps, states = {}, {}, {}
    for s in samples:                                       # sequential: no contention
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            state = await run_harness(s, retriever, answerer)
            times.append((time.perf_counter() - t) * 1000)
            for e in state.trace:
                steps.setdefault(e.step.value, []).append(e.elapsed_ms)
        per_sample[s["id"]] = times
        states[s["id"]] = state
    return first_ms, per_sample, steps, states


def sequential_baseline(samples, repeats):
    """The same rescore + evaluators, run in sequence with no harness and no
    threads (rescore.evaluate, the serving path). The comparison for the
    harness's concurrent verify step."""
    from rescore import evaluate, rescore
    times = []
    for s in samples:
        if (s["payload"].get("result") or {}).get("abstained"):
            continue
        raw = s["payload"]["llm"]["raw_content"]
        for _ in range(repeats):
            t = time.perf_counter()
            evaluate(rescore(s, raw, not s.get("dropped_by_script", False)))
            times.append((time.perf_counter() - t) * 1000)
    return times


def build(repeats):
    samples = load_samples(SAMPLES_PATH)
    recorded_doc = json.load(open(SAMPLES_PATH, encoding="utf-8"))
    first_ms, per_sample, steps, states = asyncio.run(timed_runs(samples, repeats))
    baseline = sequential_baseline(samples, repeats)

    questions = []
    for s in samples:
        st = states[s["id"]]
        last = st.attempts[-1] if st.attempts else None
        rec = s.get("recorded") or {}
        payload = s.get("payload") or {}
        questions.append({
            "id": s["id"], "kind": s["kind"], "expect": s["expect"],
            "question": payload.get("question"),
            "evaluators": ({k: v["label"] for k, v in last.evaluations.items()} if last else None),
            "harness": {"outcome": st.outcome.value,
                        "abstain_reason": st.abstain_reason.value if st.abstain_reason else None,
                        "attempts": len(st.attempts), "retried": st.retried},
            "latency_ms": summary(per_sample[s["id"]]),
            "recorded_tokens": {"model_calls": rec.get("model_calls"),
                                "total_tokens": rec.get("tokens"),
                                "prompt_tokens": payload.get("prompt_tokens") if rec.get("model_calls") else None,
                                "derived_from": rec.get("derived_from")},
        })

    names = next((list(q["evaluators"]) for q in questions if q["evaluators"]), [])
    per_evaluator = {n: {lab: sum(1 for q in questions if q["evaluators"] and q["evaluators"][n] == lab)
                         for lab in ("PASS", "FAIL", "INCONCLUSIVE", "NOT_APPLICABLE")}
                     for n in names}
    outcomes = {"accepted": sum(1 for q in questions if q["harness"]["outcome"] == "accepted"),
                "abstained": sum(1 for q in questions if q["harness"]["outcome"] == "abstained"),
                "retried": sum(1 for q in questions if q["harness"]["retried"])}
    reasons = {}
    for q in questions:
        r = q["harness"]["abstain_reason"]
        if r:
            reasons[r] = reasons.get(r, 0) + 1

    answered = [q for q in questions if q["recorded_tokens"]["model_calls"]]
    tok = [q["recorded_tokens"]["total_tokens"] for q in answered]
    ptok = [q["recorded_tokens"]["prompt_tokens"] for q in answered]
    judge = json.load(open(JUDGE_PATH, encoding="utf-8")) if os.path.exists(JUDGE_PATH) else None

    all_runs = [t for times in per_sample.values() for t in times]
    return {
        "label": "measured metrics (replay mode)",
        "generated_by": "scripts/run_metrics.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "processor": platform.processor() or None},
        "source": {"samples": "samples/recorded_outputs.json", "model": recorded_doc.get("model"),
                   "label": recorded_doc.get("label")},
        "notes": {
            "latency": ("Measured: wall-clock time of one harness run in REPLAY mode on the machine "
                        "above. Covers retrieval replay (recorded candidates), the 8 evaluators and "
                        "the harness loop. Does NOT include a model call or any network."),
            "tokens": ("Recorded, not measured here: copied from the record_samples.py run that "
                       "made each answer. Seeded transforms (s01, s03) made no call."),
            "retries": ("Replay has one recorded attempt per sample, so 'retried' is 0 by "
                        "construction. The retry path is covered by tests/test_harness.py."),
        },
        "summary": {
            "samples": len(questions),
            "harness_outcomes": outcomes,
            "abstain_reasons": reasons,
            "evaluators": per_evaluator,
            "latency_ms": {"first_run_including_imports_ms": round(first_ms, 3),
                           "all_runs": summary(all_runs),
                           "per_step": {k: summary(v) for k, v in sorted(steps.items())},
                           "sequential_rescore_evaluate_baseline": summary(baseline),
                           "repeats_per_sample": repeats,
                           "note": ("The harness verify step runs the 8 evaluators concurrently "
                                    "in worker threads. Compare it with the sequential baseline: "
                                    "for sub-millisecond work, thread hand-off costs more than it "
                                    "saves. Concurrency here is for isolation, not speed.")},
            "recorded_tokens_per_answer": {
                "answers_with_a_model_call": len(answered),
                "total_tokens": {"mean": round(statistics.mean(tok), 1), "min": min(tok), "max": max(tok)},
                "prompt_tokens": {"mean": round(statistics.mean(ptok), 1), "min": min(ptok), "max": max(ptok)},
                "sum_total_tokens": sum(tok),
            } if answered else None,
            "recorded_llm_judge": ({"label": judge["label"], "model": judge["model"],
                                    "judge_passed": judge["summary"]["judge_passed"],
                                    "judged": judge["summary"]["judged"],
                                    "total_tokens": sum((r.get("usage") or {}).get("total_tokens", 0)
                                                        for r in judge["results"])}
                                   if judge else None),
        },
        "questions": questions,
    }


def main():
    p = argparse.ArgumentParser(description="Measure the fixed set through the harness.")
    p.add_argument("--repeats", type=int, default=50, help="timed runs per sample")
    p.add_argument("--out", default=OUT_PATH)
    args = p.parse_args()
    doc = build(args.repeats)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    s = doc["summary"]
    lat = s["latency_ms"]
    print(f"[metrics] {s['samples']} samples: {s['harness_outcomes']}, abstain reasons {s['abstain_reasons']}")
    print(f"[metrics] harness run (replay, this machine): p50 {lat['all_runs']['p50_ms']} ms, "
          f"p95 {lat['all_runs']['p95_ms']} ms over {lat['all_runs']['n']} runs; "
          f"first run incl. imports {lat['first_run_including_imports_ms']} ms")
    v, b = lat["per_step"]["verify"], lat["sequential_rescore_evaluate_baseline"]
    print(f"[metrics] verify step (concurrent) p50 {v['p50_ms']} ms vs sequential "
          f"rescore+evaluate p50 {b['p50_ms']} ms")
    t = s["recorded_tokens_per_answer"]
    if t:
        print(f"[metrics] recorded tokens per answer ({t['answers_with_a_model_call']} answers): "
              f"mean {t['total_tokens']['mean']}, prompt mean {t['prompt_tokens']['mean']}")
    print(f"[metrics] wrote {os.path.relpath(args.out, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
