"""
dev_fake_live_api.py -- LOCAL DEVELOPMENT ONLY. Runs the API with live answers
ON but a FAKE model client, so the web app's live flow can be built and
tested with no key, no cost and no network:

    python scripts/dev_fake_live_api.py                  # http://127.0.0.1:8080
    python scripts/dev_fake_live_api.py --out-of-credit  # every model call fails like a drained account

The fake replays a recorded Claude answer when the question matches a
recorded sample, and otherwise returns a polite decline. Every response's
note says FAKE. Not in the Docker image (the allow-list ships only
scripts/release_gate.py).
"""
import json
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import uvicorn  # noqa: E402

from server.app import create_app  # noqa: E402
import live_answer  # noqa: E402  (server/ is on sys.path after the import above)
from live_retrieval import LiveBM25, default_path  # noqa: E402

RECORDED = {s["payload"]["question"].lower(): s["payload"]["llm"]["raw_content"]
            for s in json.load(open(os.path.join(REPO_ROOT, "samples", "recorded_outputs.json"),
                                    encoding="utf-8"))["scenarios"]
            if s["kind"] == "recorded" and (s["payload"].get("llm") or {}).get("raw_content")}
OUT_OF_CREDIT = "--out-of-credit" in sys.argv
DECLINE = json.dumps({"answer": "The retrieved records do not say.", "answerable": False,
                      "record_ids_used": [], "record_ids_excluded": [], "limitations": "FAKE local model"})


class FakeAnthropic:
    def __init__(self):
        fake = self

        class Messages:
            def count_tokens(self, model, messages, system=None):
                return types.SimpleNamespace(input_tokens=4300 if system else 3700)

            def create(self, **kw):
                if OUT_OF_CREDIT:
                    raise sys.modules["anthropic"].BadRequestError(
                        "Your credit balance is too low to access the Anthropic API.")
                judge = "system" not in kw
                q = kw["messages"][0]["content"]
                text = ('{"label": "PASS", "explanation": "FAKE judge: reads fine."}' if judge
                        else next((a for k, a in RECORDED.items() if k in q.lower()), DECLINE))
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                    stop_details=None, id="msg_fake_local",
                    usage=types.SimpleNamespace(input_tokens=250 if judge else 4300, output_tokens=180))

        fake.messages = Messages()


def main():
    sdk = types.ModuleType("anthropic")          # exception classes _call_anthropic catches
    for name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError",
                 "RateLimitError", "APIStatusError", "APIConnectionError"):
        setattr(sdk, name, type(name, (Exception,), {}))
    sys.modules["anthropic"] = sdk
    live_answer.MODE_NOTE = "FAKE MODEL (local development). " + live_answer.MODE_NOTE
    answerer = live_answer.LiveAnswerer(LiveBM25(default_path(REPO_ROOT)), live_answer.MemoryDailyCounter(50),
                                        live_answer.PerVisitor(50), client=FakeAnthropic())
    uvicorn.run(create_app(answerer=answerer), host="127.0.0.1", port=8080, log_level="warning")


if __name__ == "__main__":
    main()
