"""
demo_phoenix_evals.py -- Traces five questions through the Museum QA pipeline
into a local Arize Phoenix session, then evaluates those traces with the
deterministic verifiers and a standard LLM-as-a-judge, side by side.

    pip install -r requirements.txt
    python demo_phoenix_evals.py                     # offline: 10-record fixture + scripted model
    python demo_phoenix_evals.py --mode corpus       # full public corpus + bge-small, scripted model
    python demo_phoenix_evals.py --mode live         # full public corpus + a real model
                                                     # (ANTHROPIC_API_KEY in .env -> Claude Haiku 5.5)

Steps:
  1. Start Phoenix (phoenix.launch_app(), or attach to a running `phoenix serve`).
  2. Instrument the pipeline with OpenInference spans (src/tracing.py).
  3. Run 5 questions -- a clean answer, a zero-sources false pass, a silently
     dropped system prompt, a fabricated citation, and an out-of-corpus
     question that should abstain before any paid call.
  4. Pull the root spans back OUT of Phoenix and run the evaluators on them
     (src/phoenix_evals.py); write the verdicts back as span annotations.
  5. Print the side-by-side and the Phoenix URL.

Modes:
  fixture  No network, no credentials, no torch. Retrieval, entity resolution,
           prompt assembly, parsing, verification and confidence are the real
           pipeline code over 10 real public records (src/fixtures/); only the
           dense encoder (TF-IDF) and the model are stand-ins (see
           src/fixture_mode.py). Each question is paired with a scripted model
           response that exercises one failure mode.
  corpus   The full ~5,000-record public corpus (`python src/main.py fetch &&
           python src/main.py ingest`) and the real BM25 + bge-small retriever,
           with the same scripted model. No credentials needed; first run
           embeds the corpus (~4-5 min on CPU, then cached).
  live     data/normalized.jsonl + bge-small + a real model: the Anthropic API
           (ANTHROPIC_API_KEY, default claude-haiku-5-5) or the original proxy
           (API_KEY + MODEL_ENDPOINT). Five paid calls per run. The model
           answers for itself, so whether the false-pass trap fires depends on
           what it actually says -- the evaluators report it either way.
  auto     (default) fixture. Never live -- a key in .env must not make the
           default command spend money -- and never corpus (slow first run).
"""
import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

NORMALIZED_PATH = os.path.join(REPO_ROOT, "data", "normalized.jsonl")
REPORT_PATH = os.path.join(REPO_ROOT, "outputs", "phoenix_eval_report.json")

# (question, fixture scenario, what it demonstrates)
DEMO_QUERIES = [
    ("What year was Coney Island Beach by Reginald Marsh made?",
     "clean",
     "Clean answer with one valid citation -- everything should pass."),
    ("Are there any violations in the Louise Bourgeois Ode to My Mother records?",
     "zero_source_false_pass",
     "The trap: a confident 'No violations found' that cites zero records."),
    ("When was Omer Fast's The Casting made?",
     "dropped_system_prompt",
     "Instructions silently dropped: accurate prose, but no JSON and no citations."),
    ("When was Jeff Brouws's Railroad Landscape #33 in Pine Plains, New York made?",
     "fabricated_citation",
     "Cites a record ID that was never in the retrieved candidate set."),
    ("What is the melting point of gallium arsenide?",
     "out_of_corpus_decline",
     "Out of corpus -- should abstain before any paid call. (On the full public "
     "corpus it scores BM25 19.75, just past the 19.5 gate, and is declined instead.)"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("auto", "fixture", "corpus", "live"), default="auto")
    p.add_argument("--judge", choices=("surrogate", "live"), default="surrogate",
                   help="baseline judge: deterministic offline surrogate, or a real "
                        "LLM call per span through the same endpoint (costs budget)")
    p.add_argument("--dev", action="store_true",
                   help="live mode on the original proxy only: send X-Mode: dev "
                        "(its free sandbox budget). Ignored by the Anthropic API.")
    p.add_argument("--project", default=None, help="Phoenix project name")
    p.add_argument("--no-launch", action="store_true",
                   help="do not call phoenix.launch_app(); attach to a running server")
    p.add_argument("--exit", action="store_true",
                   help="exit when done instead of keeping the in-process Phoenix app alive")
    return p.parse_args()


def resolve_mode(requested):
    """auto never picks live: putting a key in .env must not turn the default
    command into a paid run. It also never picks corpus (slow first run)."""
    if requested != "auto":
        return requested
    from api_client import check_credentials
    if os.path.exists(NORMALIZED_PATH) and check_credentials(quiet=True):
        print("[demo] Credentials found, but auto mode stays offline -- pass "
              "--mode live to make paid calls.")
    return "fixture"


def build_retriever(mode):
    if mode == "fixture":
        from fixture_mode import load_fixture_retriever
        return load_fixture_retriever()
    if not os.path.exists(NORMALIZED_PATH):
        raise RuntimeError(f"{mode} mode needs {NORMALIZED_PATH} -- run "
                           f"`python src/main.py fetch` then `python src/main.py ingest` "
                           f"first, or use --mode fixture.")
    from retrieval import HybridRetriever, load_records
    return HybridRetriever(load_records(NORMALIZED_PATH))


def wait_for_payloads(project, url, span_ids, timeout_s=15.0):
    """Polls Phoenix until every root span from this run is queryable.
    Returns ({span_id: payload}, source) where source says whether the
    payloads came from Phoenix or from the in-process fallback."""
    from phoenix_evals import fetch_root_span_payloads
    from tracing import get_recorded_payload

    deadline = time.time() + timeout_s
    found = {}
    while time.time() < deadline:
        rows = fetch_root_span_payloads(project, url)
        if rows is None:
            break
        found = {sid: p for sid, p in rows if sid in span_ids}
        if len(found) == len(span_ids):
            return found, "phoenix"
        time.sleep(1.0)

    missing = [s for s in span_ids if s not in found]
    if missing:
        print(f"[demo] {len(missing)} span(s) not yet queryable from Phoenix -- "
              f"evaluating those from the in-process copy of the same trace payload.")
    for sid in missing:
        payload = get_recorded_payload(sid)
        if payload is not None:
            found[sid] = payload
    return found, ("phoenix" if not missing else "phoenix+fallback" if len(missing) < len(span_ids)
                   else "in-process fallback")


def _fmt(res):
    return f"{res.label}" + (f" ({res.score:.2f})" if res.score is not None else "")


def print_side_by_side(rows):
    line = "=" * 100
    print("\n" + line)
    print("SIDE-BY-SIDE: standard LLM-as-a-judge vs deterministic verifiers")
    print(line)
    for row in rows:
        print(f"\nQ{row['n']}: {row['question']}")
        print(f"    scenario : {row['why']}")
        print(f"    output   : {(row['answer_text'] or '(none)')[:110]}")
        print(f"    citations: {row['cited']}   prompt_tokens: {row['prompt_tokens']}"
              f"   confidence: {row['confidence']}")
        print(f"    LLM judge      -> {row['judge'].label:<5} {row['judge'].explanation}")
        print(f"    Deterministic  -> {row['verdict'].label:<5} "
              f"{row['verdict'].explanation[:300]}")
        for name, res in row["evals"].items():
            print(f"        {name:<34} {_fmt(res):<22} {res.explanation[:120]}")

    caught = [r for r in rows if r["judge"].label == "PASS" and r["verdict"].label == "FAIL"]
    print("\n" + line)
    print(f"Judge PASSED but deterministic verifiers FAILED: {len(caught)} of {len(rows)}")
    for r in caught:
        print(f"  - Q{r['n']}: {r['question']}")
    print(line)
    return caught


def main():
    args = parse_args()
    import reasoning
    from phoenix_evals import (DETERMINISTIC_EVALUATORS, deterministic_verdict,
                               llm_judge_surrogate, log_annotations,
                               make_live_llm_judge, run_evaluators)
    from tracing import DEFAULT_PROJECT, instrument_pipeline, start_phoenix

    mode = resolve_mode(args.mode)
    project = args.project or f"{DEFAULT_PROJECT}-{mode}"
    print(f"[demo] mode={mode}  judge={args.judge}  project={project}")

    # 1. Phoenix
    try:
        session = start_phoenix(project_name=project, launch=not args.no_launch)
    except RuntimeError as e:
        print(f"[demo] ERROR: {e}")
        return 1
    url = session["url"]
    how = {"launch_app": "launched via phoenix.launch_app()",
           "subprocess": "launched via `phoenix serve` (child process)"}.get(
        session["launched"], "attached to running server")
    print(f"[demo] Phoenix {how} at {url}")

    # 2. Pipeline + instrumentation. The scripted model is installed BEFORE
    # instrumenting so the LLM span wraps it like it would the real client.
    try:
        retriever = build_retriever(mode)
    except Exception as e:  # noqa: BLE001 -- surfaced with guidance
        print(f"[demo] ERROR building retriever: {type(e).__name__}: {e}")
        return 1
    model = None
    if mode in ("fixture", "corpus"):
        from fixture_mode import ScriptedModel
        model = ScriptedModel()
        reasoning.call_haiku = model
    if mode == "live":
        from api_client import PROVIDER, active_model, check_credentials
        if not check_credentials():
            print("[demo] ERROR: live mode needs credentials -- ANTHROPIC_API_KEY in .env "
                  "for the Anthropic API, or API_KEY + MODEL_ENDPOINT for the proxy.")
            return 1
        extra = " + 5 judge calls" if args.judge == "live" else ""
        print(f"[demo] LIVE: {len(DEMO_QUERIES)} paid call(s){extra} on {active_model()} "
              f"via {PROVIDER}")
    # Exact token counts only for a real model: the scripted one bills by the
    # character estimate, so its audit must use the same estimate.
    instrument_pipeline(exact_token_counts=(mode == "live"))

    # 3. Run the five questions
    results = []
    for n, (question, scenario, why) in enumerate(DEMO_QUERIES, start=1):
        if model is not None:
            model.scenario = scenario
        print(f"\n[demo] Q{n}: {question}")
        try:
            result = reasoning.answer_question(question, retriever, dev_mode=args.dev)
        except Exception as e:  # noqa: BLE001 -- one failed question must not end the demo
            print(f"[demo]   pipeline raised {type(e).__name__}: {e} (recorded on the span)")
            continue
        outcome = ("abstained" if result.get("abstained")
                   else f"error: {result['error']}" if result.get("error")
                   else f"answered, cited={result.get('record_ids_used')}")
        print(f"[demo]   -> {outcome}")
        results.append((n, question, why, result))

    session["tracer_provider"].force_flush()

    # 4. Evaluate the traces
    span_ids = [r["_trace"]["span_id"] for *_, r in results if r.get("_trace")]
    payloads, source = wait_for_payloads(project, url, span_ids)
    print(f"\n[demo] Evaluating {len(payloads)} root span(s) (payload source: {source})")

    judge = make_live_llm_judge(dev_mode=args.dev) if args.judge == "live" else llm_judge_surrogate
    judge_name = "llm_judge" if args.judge == "live" else "llm_judge_surrogate"
    judge_kind = "LLM" if args.judge == "live" else "CODE"

    rows, annotations = [], []
    for n, question, why, result in results:
        sid = (result.get("_trace") or {}).get("span_id")
        payload = payloads.get(sid)
        if payload is None:
            print(f"[demo] Q{n}: no trace payload found -- skipped")
            continue
        evals = run_evaluators(payload, DETERMINISTIC_EVALUATORS)
        verdict = deterministic_verdict(evals)
        judge_res = run_evaluators(payload, {judge_name: judge})[judge_name]

        for name, res in evals.items():
            annotations.append({"span_id": sid, "name": name, "label": res.label,
                                "score": res.score, "explanation": res.explanation,
                                "annotator_kind": "CODE"})
        annotations.append({"span_id": sid, "name": "deterministic_verdict",
                            "label": verdict.label, "score": verdict.score,
                            "explanation": verdict.explanation, "annotator_kind": "CODE"})
        annotations.append({"span_id": sid, "name": judge_name, "label": judge_res.label,
                            "score": judge_res.score, "explanation": judge_res.explanation,
                            "annotator_kind": judge_kind})

        res_ = payload.get("result") or {}
        conf = res_.get("confidence") or {}
        rows.append({
            "n": n, "question": question, "why": why, "span_id": sid,
            "answer_text": res_.get("answer") or res_.get("raw_content") or res_.get("reason"),
            "cited": res_.get("record_ids_used") or [],
            "prompt_tokens": payload.get("prompt_tokens"),
            "confidence": (f"{conf.get('score')} ({conf.get('label')})" if conf
                           else "none emitted"),
            "evals": evals, "verdict": verdict, "judge": judge_res,
        })

    written = log_annotations(annotations, url)
    print(f"[demo] Logged {written} span annotation(s) to Phoenix")

    caught = print_side_by_side(rows)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "mode": mode, "project": project, "payload_source": source,
            "judge": judge_name,
            "rows": [{**{k: v for k, v in r.items() if k not in ("evals", "verdict", "judge")},
                      "evals": {k: v.to_dict() for k, v in r["evals"].items()},
                      "deterministic_verdict": r["verdict"].to_dict(),
                      "judge": r["judge"].to_dict()} for r in rows],
            "judge_pass_verifier_fail": [r["n"] for r in caught],
        }, f, indent=2, ensure_ascii=False)
    print(f"[demo] Report written to {os.path.relpath(REPORT_PATH, REPO_ROOT)}")
    if mode == "live" or args.judge == "live":
        from api_client import get_tracker
        for line in get_tracker().summary().splitlines():
            print(f"[demo] {line}")

    print(f"\n  Traces + evals:  {url}/projects  (project: {project})")
    print("  Open a trace -> Annotations tab to see every evaluator's verdict.\n")

    if session["launched"] and not args.exit:
        print("Phoenix was started by this script and stops when it exits. "
              "Press Ctrl+C to stop it.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\n[demo] Stopping Phoenix.")
    stop_phoenix(session)
    return 0


def stop_phoenix(session):
    """Shuts down whatever this script started; leaves an attached server alone."""
    if session["launched"] == "launch_app":
        try:
            import phoenix as px
            px.close_app()
        except Exception as e:  # noqa: BLE001 -- best-effort shutdown
            print(f"[demo] close_app() failed: {e}")
    proc = session.get("server_process")
    if proc is not None and proc.poll() is None:
        proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
