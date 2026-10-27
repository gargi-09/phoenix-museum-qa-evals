"""
record_llm_judge.py -- ONE-TIME, LOCAL-ONLY. Runs the REAL reference-free LLM
judge (phoenix_evals.make_live_llm_judge, unchanged) over the three seeded
false-pass samples and saves whatever it decides.

    python scripts/record_llm_judge.py          # prints the plan; no calls
    python scripts/record_llm_judge.py --yes    # 3 paid calls, writes the file

Needs ANTHROPIC_API_KEY in YOUR .env (never committed, never deployed) and
samples/recorded_outputs.json from record_samples.py.

The judge sees exactly what its prompt (phoenix_evals.JUDGE_PROMPT) gives it:
the question and the answer text. No candidates, citations or token counts.
Nothing here assumes the verdict: a FAIL, an INCONCLUSIVE or a failed call is
written down as it came back.

Writes samples/recorded_llm_judge.json, labelled "recorded LLM-judge outputs".
The deployed service never imports this script and the Docker image does not
contain it.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

SAMPLES_PATH = os.path.join(REPO_ROOT, "samples", "recorded_outputs.json")
OUT_PATH = os.path.join(REPO_ROOT, "samples", "recorded_llm_judge.json")
SEEDED_IDS = ("s01", "s02", "s03")


def refuse(msg):
    print(f"[judge] REFUSING: {msg}")
    sys.exit(2)


def guard():
    if os.getenv("K_SERVICE") or os.getenv("LLM_PROVIDER", "").lower() == "disabled":
        refuse("this is the serving environment. Recording is local-only.")
    if not os.path.exists(SAMPLES_PATH):
        refuse(f"{SAMPLES_PATH} is missing. Run record_samples.py first.")
    import api_client
    if api_client.PROVIDER != "anthropic" or not api_client.check_credentials(quiet=True):
        refuse("needs ANTHROPIC_API_KEY in .env (Anthropic provider). Nothing was called.")
    return api_client


def main():
    p = argparse.ArgumentParser(description="Record the real LLM judge on the seeded samples.")
    p.add_argument("--yes", action="store_true", help="make the paid calls")
    args = p.parse_args()

    api_client = guard()
    doc = json.load(open(SAMPLES_PATH, encoding="utf-8"))
    by_id = {e["id"]: e for e in doc["scenarios"]}
    missing = [i for i in SEEDED_IDS if i not in by_id]
    if missing:
        refuse(f"samples missing from the recorded file: {missing}")
    print(f"[judge] {len(SEEDED_IDS)} judge call(s) on {api_client.active_model()}: "
          f"{', '.join(SEEDED_IDS)}.")
    if not args.yes:
        print("[judge] Dry run. Re-run with --yes to make the calls.")
        return 0

    import phoenix_evals

    # Wrapped BEFORE make_live_llm_judge binds call_haiku, only to keep the
    # judge's raw output and usage next to its parsed verdict.
    real_call, last = api_client.call_haiku, {}

    def call(messages, **kw):
        res = real_call(messages, **kw)
        last.clear()
        last.update(prompt=messages[0]["content"], raw=res.get("content"),
                    usage=res.get("usage"), error_type=res.get("error_type"))
        return res

    api_client.call_haiku = call
    judge = phoenix_evals.make_live_llm_judge(dev_mode=False)

    out = []
    for sid in SEEDED_IDS:
        e = by_id[sid]
        result = judge(e["payload"])
        det = e["python_verdicts"]["deterministic_verdict"]
        out.append({
            "id": sid, "expect": e["expect"], "question": e["payload"]["question"],
            "judge": {"label": result.label, "score": result.score,
                      "explanation": result.explanation},
            "judge_prompt": last.get("prompt"), "judge_raw_output": last.get("raw"),
            "usage": last.get("usage"), "error_type": last.get("error_type"),
            "deterministic_verdict": det["label"],
            "rule_based_surrogate": e["python_verdicts"]["llm_judge_surrogate"]["label"],
        })
        print(f"[judge] {sid}: LLM judge={result.label:<13} deterministic={det['label']:<6} "
              f"{result.explanation[:90]}")

    approved = [r["id"] for r in out if r["judge"]["label"] == "PASS"]
    record = {
        "label": "recorded LLM-judge outputs",
        "notice": ("The real reference-free LLM judge (phoenix_evals.make_live_llm_judge, "
                   "prompt phoenix_evals.JUDGE_PROMPT) run once, locally, on the three "
                   "seeded false-pass samples. Verdicts are as returned; none is edited."),
        "generated_by": "scripts/record_llm_judge.py",
        "model": api_client.active_model(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "judged_from": "samples/recorded_outputs.json",
        "summary": {"judged": len(out), "judge_passed": approved,
                    "judge_passed_count": len(approved)},
        "results": out,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    print(f"[judge] wrote {os.path.relpath(OUT_PATH, REPO_ROOT)}")
    print(f"[judge] the LLM judge passed {len(approved)}/{len(out)} seeded false passes: "
          f"{approved or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
