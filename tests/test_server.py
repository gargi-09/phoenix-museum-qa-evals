"""API behaviour: health, samples-only, CORS, rate limiting, body limits, and
live evaluator results matching the verdicts recorded alongside each sample."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from conftest import sample_files
from server.app import SAMPLES_ONLY, create_app

GH_PAGES = "https://gargi-09.github.io"


def client(path=None, rate=1000):
    return TestClient(create_app(samples_path=path, rate_limit_per_minute=rate))


# --- health -----------------------------------------------------------------

def test_health_is_fast_and_reports_disabled_model_calls():
    c = client()
    t0 = time.perf_counter()
    r = c.get("/health")
    elapsed = time.perf_counter() - t0
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_calls"] == "disabled"
    assert body["samples"] > 0
    assert elapsed < 0.5


# --- samples only -------------------------------------------------------------

def test_free_text_is_refused_everywhere():
    c = client()
    for method, url, kw in [("get", "/ask?q=who painted it", {}),
                            ("post", "/query", {"json": {"q": "x"}}),
                            ("post", "/evaluate", {"json": {"question": "what is this?"}}),
                            ("post", "/evaluate", {"json": {"sample_id": "q01", "query": "x"}})]:
        r = getattr(c, method)(url, **kw)
        assert r.status_code == 400, (url, r.status_code)
        assert SAMPLES_ONLY in r.json()["detail"]


def test_unknown_sample_and_unknown_route_mention_samples_only():
    c = client()
    for url in ("/samples/not-a-sample", "/anything-else"):
        r = c.get(url)
        assert r.status_code == 404 and SAMPLES_ONLY in r.json()["detail"]
    r = c.post("/evaluate", json={"sample_id": "not-a-sample"})
    assert r.status_code == 404 and SAMPLES_ONLY in r.json()["detail"]


def test_samples_listing_shape():
    body = client().get("/samples").json()
    assert body["data_label"]
    for s in body["samples"]:
        assert {"id", "kind", "question", "abstained"} <= set(s)


# --- live evaluation ------------------------------------------------------------

@pytest.mark.parametrize("path", sample_files())
def test_live_evaluation_matches_recorded_verdicts(path):
    """Every evaluator, computed live by the API on the recorded output, must
    equal the verdict stored with the sample when it was recorded."""
    doc = json.load(open(path, encoding="utf-8"))
    c = client(path)
    checked = 0
    for s in doc["scenarios"]:
        for r in (c.get(f"/samples/{s['id']}").json(),
                  c.post("/evaluate", json={"sample_id": s["id"]}).json()):
            live = dict(r["evaluations"])
            live["deterministic_verdict"] = r["deterministic_verdict"]
            live["llm_judge_surrogate"] = r["llm_judge_surrogate"]
            for name, want in s["python_verdicts"].items():
                assert (live[name]["label"], live[name]["score"]) == (want["label"], want["score"]), \
                    (s["id"], name)
                checked += 1
    assert checked > 0


def test_edited_answer_is_rescored_live():
    c = client()
    samples = c.get("/samples").json()["samples"]
    target = next(s for s in samples if not s["abstained"])
    detail = c.get(f"/samples/{target['id']}").json()
    cands = [x["id"] for x in detail["retrieval"]["candidates"]]
    fake = "aic-0000000"
    assert fake not in cands
    edited = json.dumps({"answer": "It was made in 1935.", "answerable": True,
                         "record_ids_used": [fake], "record_ids_excluded": [],
                         "limitations": ""})
    r = c.post("/evaluate", json={"sample_id": target["id"], "answer": edited}).json()
    assert r["source"] == "edited"
    assert r["evaluations"]["citation_groundedness"]["label"] == "FAIL"
    assert r["deterministic_verdict"]["label"] == "FAIL"
    assert "No model was called" in r["notice"]


def test_toggling_system_prompt_delivery_drives_prompt_integrity():
    c = client()
    s = next(s for s in c.get("/samples").json()["samples"] if not s["abstained"])
    off = c.post("/evaluate", json={"sample_id": s["id"], "system_prompt_delivered": False}).json()
    on = c.post("/evaluate", json={"sample_id": s["id"], "system_prompt_delivered": True}).json()
    assert off["evaluations"]["prompt_integrity"]["label"] == "FAIL"
    assert on["evaluations"]["prompt_integrity"]["label"] == "PASS"


# --- CORS -----------------------------------------------------------------------

def preflight(c, origin):
    return c.options("/evaluate", headers={"Origin": origin,
                                          "Access-Control-Request-Method": "POST",
                                          "Access-Control-Request-Headers": "content-type"})


def test_cors_allows_github_pages_and_localhost_only():
    c = client()
    for origin in (GH_PAGES, "http://localhost:8765", "http://127.0.0.1:5500"):
        r = preflight(c, origin)
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == origin
    for origin in ("https://evil.example", "https://gargi-09.github.io.evil.example",
                   "http://localhost.evil.example", "https://localhost:8765"):
        r = preflight(c, origin)
        assert r.headers.get("access-control-allow-origin") is None, origin
        r = c.get("/health", headers={"Origin": origin})
        assert r.headers.get("access-control-allow-origin") is None, origin


def test_cors_never_uses_a_wildcard():
    c = client()
    for origin in (GH_PAGES, "https://evil.example"):
        r = c.get("/samples", headers={"Origin": origin})
        assert r.headers.get("access-control-allow-origin") != "*"


# --- rate limiting and size limits ----------------------------------------------

def test_rate_limit_per_ip_returns_429_with_cors():
    c = client(rate=3)
    hdr = {"X-Forwarded-For": "203.0.113.7", "Origin": GH_PAGES}
    codes = [c.get("/samples", headers=hdr).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    r = c.get("/samples", headers=hdr)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert r.headers.get("access-control-allow-origin") == GH_PAGES
    # Another client is unaffected, and /health is never limited.
    assert c.get("/samples", headers={"X-Forwarded-For": "198.51.100.2"}).status_code == 200
    assert c.get("/health", headers=hdr).status_code == 200


def test_default_rate_limit_is_30_per_minute():
    c = TestClient(create_app())
    hdr = {"X-Forwarded-For": "192.0.2.55"}
    codes = [c.get("/samples", headers=hdr).status_code for _ in range(31)]
    assert codes.count(200) == 30 and codes[-1] == 429


def test_request_body_size_limit():
    c = client()
    big = {"sample_id": "q01", "answer": "x" * 20000}
    r = c.post("/evaluate", content=json.dumps(big), headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_answer_length_limit():
    c = client()
    r = c.post("/evaluate", json={"sample_id": "q01", "answer": "x" * 9000})
    assert r.status_code in (413, 422)
