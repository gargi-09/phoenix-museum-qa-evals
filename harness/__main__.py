"""python -m harness [--samples PATH] [--json]

Runs the harness in REPLAY mode over the fixed sample set and prints each
outcome. No model is called: model calls are disabled before any pipeline
import.
"""
import argparse
import asyncio
import json
import os
import sys

os.environ["LLM_PROVIDER"] = "disabled"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from harness.loop import run_all  # noqa: E402
from harness.replay import ReplayAnswerer, ReplayRetriever, load_samples  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Run the harness in replay mode.")
    p.add_argument("--samples", default=os.path.join(REPO_ROOT, "samples", "recorded_outputs.json"))
    p.add_argument("--json", action="store_true", help="print full states as JSON")
    args = p.parse_args()

    states = asyncio.run(run_all(load_samples(args.samples), ReplayRetriever(), ReplayAnswerer()))
    if args.json:
        print(json.dumps([s.to_dict() for s in states], indent=1, default=str))
        return 0
    for s in states:
        reason = f" ({s.abstain_reason.value})" if s.abstain_reason else ""
        verdict = s.attempts[-1].verdict if s.attempts else "-"
        print(f"{s.sample_id:<4} {s.outcome.value:<9}{reason:<32} attempts={len(s.attempts)} "
              f"verdict={verdict}")
    counts = {o: sum(1 for s in states if s.outcome.value == o) for o in ("accepted", "abstained")}
    print(f"accepted {counts['accepted']}, abstained {counts['abstained']}, "
          f"retried {sum(s.retried for s in states)} (replay has one recorded attempt per sample)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
