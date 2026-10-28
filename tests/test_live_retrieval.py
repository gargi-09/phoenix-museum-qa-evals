"""POST /retrieve: live BM25 over the committed corpus. Its scores must be the
pipeline's own BM25 scores (the recorded gate values for q10 and q11), and it
must never generate an answer.
"""
import gzip
import json
import os

import pytest
from fastapi.testclient import TestClient

from conftest import REPO_ROOT

CORPUS = os.path.join(REPO_ROOT, "samples", "corpus.jsonl.gz")
pytestmark = pytest.mark.skipif(not os.path.exists(CORPUS), reason="needs samples/corpus.jsonl.gz")


@pytest.fixture(scope="module")
def client():
    from server.app import create_app
    return TestClient(create_app(rate_limit_per_minute=1000))


def post(client, question):
    return client.post("/retrieve", json={"question": question})


def test_corpus_is_the_full_slim_corpus():
    with gzip.open(CORPUS, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 5010
    assert {r["id"][:4] for r in rows} == {"cma-", "aic-"}
    # The full normalized record: the live-answer path builds context blocks from it.
    assert {"id", "title", "artist", "years", "medium", "dimensions", "classification",
            "description"} <= set(rows[0])


def test_scores_match_the_recorded_pipeline_gate(client):
    # samples.json records these from the real hybrid retriever's BM25 half.
    q11 = post(client, "What is the capital of Mongolia?").json()
    assert q11["gate"]["bm25_max"] == 17.22 and q11["gate"]["abstained"] is True
    q10 = post(client, "What is the melting point of gallium arsenide?").json()
    assert q10["gate"]["bm25_max"] == 19.75 and q10["gate"]["abstained"] is False


def test_a_grounded_question_finds_its_record(client):
    r = post(client, "What year was Coney Island Beach by Reginald Marsh made?").json()
    assert r["candidates"][0]["id"] == "cma-117571"
    assert r["recorded_sample_id"] == "q01"
    assert r["records_searched"] == 5010 and len(r["candidates"]) == 10
    assert r["mode"] == "live BM25 only" and "dense" in r["note"]


def test_never_answers(client):
    r = post(client, "Tell me about Louise Bourgeois").json()
    assert "answer" not in r and "llm" not in r
    assert "never makes" in r["note"]


def test_aic_descriptions_carry_their_licence(client):
    r = post(client, "Art Institute of Chicago painting").json()
    for c in r["candidates"]:
        if c["id"].startswith("aic-") and c["snippet"]:
            assert "CC BY 4.0" in c["attribution"]
        if c["id"].startswith("cma-"):
            assert c["attribution"] is None


@pytest.mark.parametrize("body,status", [
    ({"question": ""}, 422),
    ({"question": "   "}, 422),
    ({"question": "x" * 301}, 422),
    ({"question": "ok", "answer": "smuggled"}, 400),
    ({}, 422),
])
def test_input_limits(client, body, status):
    r = client.post("/retrieve", json=body)
    assert r.status_code == status, r.text


def test_free_text_answer_routes_are_still_refused(client):
    assert client.get("/ask?q=hi").status_code == 400
    assert client.post("/answer", json={"question": "hi"}).status_code == 400


def test_health_reports_live_retrieval(client):
    h = client.get("/health").json()
    assert h["live_retrieval"] == "bm25" and h["records"] == 5010 and h["model_calls"] == "disabled"


def test_retrieve_is_rate_limited_and_cors_scoped():
    from server.app import create_app
    c = TestClient(create_app(rate_limit_per_minute=2))
    hdr = {"Origin": "https://gargi-09.github.io"}
    codes = [c.post("/retrieve", json={"question": "Marsh"}, headers=hdr).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = c.post("/retrieve", json={"question": "Marsh"}, headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
