"""Live answers (server/live_answer.py) and the live release gate
(server/gate_runner.py), with a FAKE Anthropic client: no key, no network,
no real model call. Checks the safeguards fail closed and that the verdicts
come from the real pipeline pieces.
"""
import io
import json
import os
import sys
import types
import urllib.error

import pytest
from fastapi.testclient import TestClient

from conftest import RECORDED, REPO_ROOT

CORPUS = os.path.join(REPO_ROOT, "samples", "corpus.jsonl.gz")
pytestmark = pytest.mark.skipif(not (os.path.exists(CORPUS) and os.path.exists(RECORDED)),
                                reason="needs the corpus and recorded samples")

Q01 = "What year was Coney Island Beach by Reginald Marsh made?"


def recorded_raw(sid):
    doc = json.load(open(RECORDED, encoding="utf-8"))
    return next(s for s in doc["scenarios"] if s["id"] == sid)["payload"]["llm"]["raw_content"]


class FakeClient:
    """Stands in for anthropic.Anthropic(). Bills the system prompt unless
    drop_system_in_billing, to exercise the prompt-token audit."""

    def __init__(self, answer, judge='{"label": "PASS", "explanation": "Looks right."}',
                 drop_system_in_billing=False):
        self.answer, self.judge, self.drop = answer, judge, drop_system_in_billing
        self.creates, self.counts = [], []
        outer = self

        class Messages:
            def count_tokens(self, model, messages, system=None):
                outer.counts.append(bool(system))
                return types.SimpleNamespace(input_tokens=4300 if system else 3700)

            def create(self, **kw):
                outer.creates.append(kw)
                is_judge = "system" not in kw
                billed = 250 if is_judge else (3700 if outer.drop else 4300)
                text = outer.judge if is_judge else outer.answer
                return types.SimpleNamespace(
                    content=[types.SimpleNamespace(type="text", text=text)], stop_reason="end_turn",
                    stop_details=None, id="msg_fake",
                    usage=types.SimpleNamespace(input_tokens=billed, output_tokens=180))

        self.messages = Messages()


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    """api_client._call_anthropic imports anthropic for its exception classes."""
    fake = types.ModuleType("anthropic")
    for name in ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError",
                 "RateLimitError", "APIStatusError", "APIConnectionError"):
        setattr(fake, name, type(name, (Exception,), {}))
    monkeypatch.setitem(sys.modules, "anthropic", fake)


def make(client_obj, cap=50, per_ip=5):
    # Import through the app's own path (server/ on sys.path), so the
    # exception classes the routes catch are the ones raised here.
    from server.app import create_app
    from live_answer import LiveAnswerer, MemoryDailyCounter, PerVisitor
    from live_retrieval import LiveBM25
    live = LiveBM25(CORPUS)
    answerer = LiveAnswerer(live, MemoryDailyCounter(cap), PerVisitor(per_ip), client=client_obj)
    return TestClient(create_app(rate_limit_per_minute=1000, answerer=answerer)), answerer


def ask(c, q):
    return c.post("/live/answer", json={"question": q})


# --- the live answer path ---------------------------------------------------

def test_a_grounded_live_answer_is_verified_and_accepted():
    fake = FakeClient(recorded_raw("q01"))
    c, _ = make(fake)
    r = ask(c, Q01).json()
    assert r["outcome"] == "accepted" and r["model_called"] is True
    assert r["deterministic_verdict"]["label"] == "PASS"
    assert r["llm_judge"]["label"] == "PASS"
    assert len(r["evaluations"]) == 8
    assert [s["step"] for s in r["trace"]] == ["retrieve", "gate", "quota", "build_prompt",
                                               "count_tokens", "model_call", "verify", "llm_judge", "decide"]
    assert r["prompt_audit"]["status"] == "ok"
    assert r["usage"]["answer"]["prompt_tokens"] == 4300 and r["usage"]["estimated_cost_usd"] > 0
    assert r["quota_used"] == 1
    # One answer call with the instruction block, one judge call without it.
    assert len(fake.creates) == 2 and "system" in fake.creates[0] and "system" not in fake.creates[1]


def test_the_audit_uses_the_real_billed_count():
    # The endpoint bills as if the instructions never arrived.
    c, _ = make(FakeClient(recorded_raw("q01"), drop_system_in_billing=True))
    r = ask(c, Q01).json()
    assert r["evaluations"]["prompt_integrity"]["label"] == "FAIL"
    assert r["outcome"] == "abstained" and r["abstain_reason"] == "verification_failed"


def test_an_unsupported_answer_fools_the_judge_not_the_verifier():
    parsed = json.loads(recorded_raw("q01"))
    parsed["record_ids_used"] = []
    c, _ = make(FakeClient(json.dumps(parsed)))
    r = ask(c, Q01).json()
    assert r["llm_judge"]["label"] == "PASS"
    assert r["deterministic_verdict"]["label"] == "FAIL" and r["outcome"] == "abstained"


def test_the_gate_stops_off_topic_questions_before_any_paid_call():
    fake = FakeClient(recorded_raw("q01"))
    c, answerer = make(fake)
    r = ask(c, "What is the capital of Mongolia?").json()
    assert r["outcome"] == "abstained" and r["abstain_reason"] == "retrieval_gate"
    assert r["model_called"] is False and fake.creates == [] and fake.counts == []
    assert answerer.counter.used_today() == 0            # costs no quota


def test_the_daily_cap_is_hard():
    fake = FakeClient(recorded_raw("q01"))
    c, _ = make(fake, cap=1)
    assert ask(c, Q01).status_code == 200
    r = ask(c, Q01)
    assert r.status_code == 429 and "today's limit" in r.json()["detail"]
    assert len(fake.creates) == 2                        # only the first answer (+ its judge) ran


def test_the_per_visitor_cap():
    c, _ = make(FakeClient(recorded_raw("q01")), per_ip=1)
    assert ask(c, Q01).status_code == 200
    r = ask(c, Q01)
    assert r.status_code == 429 and "hourly limit" in r.json()["detail"]


def test_an_unreachable_counter_means_no_call():
    from live_answer import QuotaUnavailable
    fake = FakeClient(recorded_raw("q01"))
    c, answerer = make(fake)

    def broken():
        raise QuotaUnavailable("down")

    answerer.counter.reserve = broken
    r = ask(c, Q01)
    assert r.status_code == 503 and fake.creates == []


@pytest.mark.parametrize("body", [{"question": ""}, {"question": "x" * 301}, {"question": "hi", "system": "x"}])
def test_live_input_limits(body):
    c, _ = make(FakeClient("{}"))
    assert c.post("/live/answer", json=body).status_code in (400, 422)


def test_status_reports_the_quota():
    c, _ = make(FakeClient(recorded_raw("q01")), cap=50, per_ip=5)
    ask(c, Q01)
    st = c.get("/live/status").json()
    assert st["enabled"] and st["daily_cap"] == 50 and st["remaining_today"] == 49
    assert st["visitor_remaining"] == 4


# --- off by default, fails closed --------------------------------------------

def test_live_answers_are_off_by_default():
    from server.app import create_app
    c = TestClient(create_app(rate_limit_per_minute=1000))
    assert c.get("/live/status").json()["enabled"] is False
    assert ask(c, Q01).status_code == 503
    assert c.get("/health").json()["live_answers"] == "off"


def test_from_env_fails_closed(monkeypatch):
    import server.app  # noqa: F401  (puts server/ on sys.path)
    from live_answer import from_env
    monkeypatch.setenv("LIVE_ANSWERS", "on")
    monkeypatch.delenv("COUNTER_BUCKET", raising=False)
    monkeypatch.delenv("LIVE_ALLOW_MEMORY_COUNTER", raising=False)
    assert from_env(object()) is None                    # no durable counter, no live answers
    monkeypatch.setenv("LIVE_ANSWERS", "off")
    monkeypatch.setenv("COUNTER_BUCKET", "b")
    assert from_env(object()) is None


# --- the durable counter -------------------------------------------------------

class FakeGcs:
    """In-memory stand-in for the two GCS JSON-API calls the counter makes."""

    def __init__(self, race_once=False, down=False):
        self.obj, self.gen, self.race_once, self.down = None, 0, race_once, down

    def __call__(self, req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        if "metadata.google.internal" in url:
            return io.BytesIO(json.dumps({"access_token": "t", "expires_in": 3600}).encode())
        if self.down:
            raise urllib.error.URLError("unreachable")
        if req.get_method() == "GET":
            if self.obj is None:
                raise urllib.error.HTTPError(url, 404, "nf", {}, None)
            r = io.BytesIO(json.dumps(self.obj).encode())
            r.headers = {"x-goog-generation": str(self.gen)}
            return _ctx(r)
        want = int(url.split("ifGenerationMatch=")[1])
        if self.race_once:
            self.race_once = False
            self.gen += 1
            self.obj = {"used": (self.obj or {"used": 0})["used"] + 1}
            raise urllib.error.HTTPError(url, 412, "precondition", {}, None)
        assert want == self.gen
        self.obj, self.gen = json.loads(req.data), self.gen + 1
        return _ctx(io.BytesIO(b"{}"))


class _ctx:
    def __init__(self, f):
        self.f, self.headers = f, getattr(f, "headers", {})

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, *a):
        return self.f.read(*a)


def test_gcs_counter_increments_and_caps(monkeypatch):
    import server.app  # noqa: F401
    import live_answer as la
    gcs = FakeGcs()
    monkeypatch.setattr(la.urllib.request, "urlopen", lambda req, timeout=None: _wrap(gcs(req, timeout)))
    c = la.GcsDailyCounter("bucket", cap=2)
    assert c.reserve() == 1 and c.reserve() == 2
    with pytest.raises(la.QuotaExhausted):
        c.reserve()


def test_gcs_counter_survives_a_race(monkeypatch):
    import server.app  # noqa: F401
    import live_answer as la
    gcs = FakeGcs(race_once=True)
    monkeypatch.setattr(la.urllib.request, "urlopen", lambda req, timeout=None: _wrap(gcs(req, timeout)))
    assert la.GcsDailyCounter("bucket", cap=5).reserve() == 2      # the other instance took 1


def test_gcs_counter_fails_closed(monkeypatch):
    import server.app  # noqa: F401
    import live_answer as la
    gcs = FakeGcs(down=True)
    monkeypatch.setattr(la.urllib.request, "urlopen", lambda req, timeout=None: _wrap(gcs(req, timeout)))
    with pytest.raises(la.QuotaUnavailable):
        la.GcsDailyCounter("bucket", cap=5).reserve()


def _wrap(r):
    return r if hasattr(r, "__enter__") else _ctx(r)


# --- the live release gate ------------------------------------------------------

@pytest.fixture(scope="module")
def gate_client():
    from server.app import create_app
    return TestClient(create_app(rate_limit_per_minute=1000))


def test_gate_passes_on_the_committed_answers(gate_client):
    r = gate_client.post("/gate/run", json={}).json()
    assert r["release"] == "OK" and all(c["ok"] for c in r["checks"])


@pytest.mark.parametrize("brk,expect", [
    ("approve_seeded", "seeded false passes approved: 1/3"),
    ("drop_known_good", "known-good pass rate: 5/7"),
    ("hide_dropped_prompt", "identity regression"),
])
def test_each_break_blocks_the_release(gate_client, brk, expect):
    r = gate_client.post("/gate/run", json={"break": brk}).json()
    assert r["release"] == "BLOCKED"
    assert any((not c["ok"]) and expect in c["message"] for c in r["checks"])


def test_gate_rejects_unknown_breaks(gate_client):
    assert gate_client.post("/gate/run", json={"break": "rm -rf"}).status_code == 422
