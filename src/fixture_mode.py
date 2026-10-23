"""
fixture_mode.py -- Offline stand-ins so the Phoenix demo runs with no corpus,
no embedding model and no API credentials.

WHAT IS REAL in fixture mode: context assembly, entity resolution,
contradiction detection, prompt construction, response parsing, all four
verification checks, confidence scoring, provenance -- every line of
reasoning.answer_question() runs unchanged. Tracing and the Phoenix
evaluators see exactly what they would see on a live run.

WHAT IS FAKED, and only this:
  1. The corpus size -- src/fixtures/museum_fixture.jsonl is 10 REAL public
     records (the seed works the demo questions are about), fetched from the
     CMA and AIC public APIs and normalized by ingest.py. Rebuild it with
     `python src/fetch_public_corpus.py --fixture`. Nothing in it is typed by
     hand; it is just small.
  2. The dense half of retrieval -- TF-IDF cosine over word 1-2 grams instead
     of bge-small-en-v1.5, so torch is not needed. Still fused with BM25 by
     the same Reciprocal Rank Fusion, so the candidate set is still a
     sparse+dense fusion -- just with a weaker dense signal.
  3. The model -- every ID a script cites (except the deliberately fabricated
     one) is a seed record, so the same scripts also run against the full
     public corpus in demo_phoenix_evals.py's `corpus` mode. Scripted
     responses (SCRIPTED_RESPONSES), one per demo
     question, each built to exercise one failure mode. prompt_tokens is
     computed from the prompt that was ACTUALLY built for that question, so
     the token audit is measuring a real prompt, not a made-up number.

The same pattern as mock_reasoning_test.py, packaged for reuse.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api_client import detect_truncation, estimate_tokens_preflight
from retrieval import load_records, tokenize

FIXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fixtures", "museum_fixture.jsonl")

# HybridRetriever's thresholds (0.35 dense, 19.5 BM25) were calibrated on the
# 5,000-record corpus. BM25 magnitudes scale with corpus size through IDF, so
# on 10 records every question would abstain. These are placed for THIS
# fixture only, by the same method: between the best out-of-corpus score and
# the worst in-corpus score of the demo questions, as printed by
# `python src/fixture_mode.py`. Measured on the public-records fixture:
# out-of-corpus max 3.52 (gallium arsenide), in-corpus min 8.47 (Coney
# Island) -> 6.0 at the midpoint. Re-run that after rebuilding the fixture.
FIXTURE_BM25_THRESHOLD = 6.0
FIXTURE_DENSE_THRESHOLD = 0.05


class FixtureRetriever:
    """BM25 + TF-IDF, fused by RRF. Same retrieve() contract as
    retrieval.HybridRetriever: (candidates, abstain, best_similarity, signals).
    """

    def __init__(self, records, bm25_threshold=FIXTURE_BM25_THRESHOLD,
                 dense_threshold=FIXTURE_DENSE_THRESHOLD):
        from rank_bm25 import BM25Okapi
        from sklearn.feature_extraction.text import TfidfVectorizer
        from retrieval import build_passage_text

        self.records = records
        self.ids = [r["id"] for r in records]
        self.passages = [build_passage_text(r) for r in records]
        self.bm25 = BM25Okapi([tokenize(p) for p in self.passages])
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                          stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.passages)
        self.bm25_threshold = bm25_threshold
        self.dense_threshold = dense_threshold

    def retrieve(self, question, top_k=10, rrf_k=60):
        bm25_scores = np.asarray(self.bm25.get_scores(tokenize(question)))
        sims = (self.matrix @ self.vectorizer.transform([question]).T).toarray().ravel()

        rrf = {}
        for order in (np.argsort(bm25_scores)[::-1], np.argsort(sims)[::-1]):
            for rank, idx in enumerate(order, start=1):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        fused = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:top_k]

        best_sim = float(sims.max()) if len(sims) else 0.0
        bm25_max = float(bm25_scores.max()) if len(bm25_scores) else 0.0
        dense_below = best_sim < self.dense_threshold
        bm25_below = bm25_max < self.bm25_threshold

        reasons = []
        if dense_below:
            reasons.append(f"best TF-IDF similarity {best_sim:.3f} is below the "
                           f"fixture threshold {self.dense_threshold}")
        if bm25_below:
            reasons.append(f"best BM25 score {bm25_max:.2f} is below the fixture "
                           f"threshold {self.bm25_threshold} -- no record shares "
                           f"enough distinctive vocabulary with this question")

        signals = {
            "dense_similarity": best_sim,
            "bm25_max": bm25_max,
            "dense_threshold": self.dense_threshold,
            "bm25_threshold": self.bm25_threshold,
            "dense_below_threshold": dense_below,
            "bm25_below_threshold": bm25_below,
            "abstain_reasons": reasons,
            "retriever": "fixture (BM25 + TF-IDF, RRF)",
        }
        candidates = [{"id": self.ids[i], "record": self.records[i],
                       "rrf_score": rrf[i], "dense_similarity": float(sims[i])}
                      for i in fused]
        return candidates, dense_below or bm25_below, best_sim, signals


def load_fixture_retriever(path=FIXTURE_PATH):
    records = load_records(path)
    if not records:
        raise RuntimeError(f"Fixture corpus is empty or unreadable: {path}")
    return FixtureRetriever(records)


# ---------------------------------------------------------------------------
# Scripted model
# ---------------------------------------------------------------------------

# Each response is what the model "returns" for that question. `drop_system`
# simulates the endpoint behaviour measured in api_client.py's header: the
# instructions are discarded with a 200 OK and are not billed as input.
SCRIPTED_RESPONSES = {
    "clean": {
        "content": {
            "answer": "According to the museum records, Coney Island Beach by "
                      "Reginald Marsh was made in 1935.",
            "answerable": True,
            "record_ids_used": ["cma-117571"],
            "record_ids_excluded": [
                {"id": "cma-171726", "reason": "Leon Levinstein photograph, not the Marsh work"},
                {"id": "cma-157127", "reason": "Andreas Feininger photograph, not the Marsh work"},
                {"id": "aic-59820", "reason": "different Reginald Marsh work (Tattoo and Haircut)"},
            ],
            "limitations": "",
        },
    },
    "zero_source_false_pass": {
        # The trap: a confident, fluent "all clear" that cites nothing. Every
        # per-citation check has nothing to examine, so a naive pass-rate
        # reads it as 100% clean.
        "content": {
            "answer": "No violations found. The Ode to My Mother records are "
                      "consistent across both institutions, and every field is "
                      "complete and verified.",
            "answerable": True,
            "record_ids_used": [],
            "record_ids_excluded": [],
            "limitations": "",
        },
    },
    "dropped_system_prompt": {
        # Instructions never arrived, so: no JSON contract and no citations.
        # The prose is accurate, which is exactly why a judge passes it.
        "drop_system": True,
        "raw": "Omer Fast's The Casting was made in 2007. It is a four-channel "
               "video installation interweaving a soldier's recollections with "
               "staged tableaux.",
    },
    "out_of_corpus_decline": {
        # Only reached when the abstention gate lets an off-topic question
        # through. Measured: on the public corpus "melting point of gallium
        # arsenide" scores BM25 19.75, just over the 19.5 floor, so it reaches
        # the model in `corpus` mode (the fixture still abstains). A correct
        # model declines; the evaluators should read that as a decline, not a
        # false pass.
        "content": {
            "answer": "The retrieved museum records do not state the melting point "
                      "of gallium arsenide; they describe artworks, not material "
                      "properties.",
            "answerable": False,
            "record_ids_used": [],
            "record_ids_excluded": [],
            "limitations": "Question is outside the scope of the corpus.",
        },
    },
    "fabricated_citation": {
        "content": {
            # aic-99999 is the fabrication: a real-looking ID that is never in
            # the retrieved candidate set for this question.
            "answer": "Jeff Brouws's Railroad Landscape #33 in Pine Plains, New York "
                      "was made in 2009 and printed in 2010, according to the "
                      "Cleveland Museum of Art record and a matching exhibition record.",
            "answerable": True,
            "record_ids_used": ["cma-169223", "aic-99999"],
            "record_ids_excluded": [],
            "limitations": "",
        },
    },
}


class ScriptedModel:
    """Drop-in for api_client.call_haiku. Set .scenario before each question."""

    def __init__(self):
        self.scenario = None
        self.call_seq = 0

    def __call__(self, messages, max_tokens=1024, dev_mode=False, stage="unspecified",
                 expect_json=False, system=None, temperature=None, **kwargs):
        spec = SCRIPTED_RESPONSES.get(self.scenario)
        if spec is None:
            raise RuntimeError(f"No scripted response for scenario {self.scenario!r}")

        content = spec["raw"] if "raw" in spec else json.dumps(spec["content"])
        billed = estimate_tokens_preflight(messages)
        if system and not spec.get("drop_system"):
            billed += estimate_tokens_preflight(system)
        completion = max(1, len(content) // 4)
        usage = {"prompt_tokens": billed, "completion_tokens": completion,
                 "total_tokens": billed + completion}

        # The REAL truncation detector, which call_haiku() would have run.
        # Without it a scripted response that stops mid-JSON scored
        # not_truncated=PASS -- found when the dashboard's JS port, which does
        # run detection, disagreed with this fixture. A scripted retry returns
        # the same text, so a truncated response stays truncated, exactly as
        # call_haiku()'s single retry would leave it.
        is_truncated, reasons = detect_truncation(content, usage, max_tokens,
                                                  expect_json=expect_json)

        self.call_seq += 1
        return {
            "content": content,
            "usage": usage,
            "is_truncated": is_truncated,
            "truncation_reasons": reasons,
            "error_type": None,
            "should_retry": False,
            "call_seq": self.call_seq,
            "budget_remaining": None,
            "prior_truncation": ({"attempt": 1, "call_seq": self.call_seq,
                                  "reasons": reasons, "usage": usage}
                                 if is_truncated else None),
            "truncation_recovered": False,
        }


if __name__ == "__main__":
    # Threshold calibration aid: prints both signals for each demo question.
    retriever = load_fixture_retriever()
    for q in sys.argv[1:] or [
        "What year was Coney Island Beach by Reginald Marsh made?",
        "Are there any violations in the Louise Bourgeois Ode to My Mother records?",
        "When was Omer Fast's The Casting made?",
        "When was Jeff Brouws's Railroad Landscape #33 in Pine Plains, New York made?",
        "What is the melting point of gallium arsenide?",
        "What is the capital of Mongolia?",
    ]:
        _, abstain, best, sig = retriever.retrieve(q)
        print(f"bm25={sig['bm25_max']:6.2f} tfidf={best:.3f} abstain={abstain}  {q}")
