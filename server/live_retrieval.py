"""Live BM25 retrieval over the full corpus, for the public demo.

This is the lexical half of the pipeline's hybrid retriever, and its BM25
abstention floor, run live on a visitor's question. It is NOT the whole
retriever: the dense half (bge-small) needs the embedding model (torch),
which the public service does not ship, so there is no RRF fusion and no
dense gate here. The response says so.

No model is called. The question is scored and discarded; nothing logs it.
"""
import gzip
import json
import os
import time

import numpy as np
from rank_bm25 import BM25Okapi

from retrieval import BM25_ABSTENTION_THRESHOLD, build_passage_text, tokenize

AIC_ATTRIBUTION = ("Description (c) the Art Institute of Chicago, CC BY 4.0 "
                   "(https://creativecommons.org/licenses/by/4.0/).")
MODE = "live BM25 only"
NOTE = ("Live: the BM25 half of the hybrid retriever and its abstention floor "
        f"({BM25_ABSTENTION_THRESHOLD}), over every record. Not run here: the dense "
        "half (bge-small) and RRF fusion, which need the embedding model, so the "
        "ranking can differ from the full pipeline's. No answer is generated: that "
        "needs a model call, which this public demo never makes.")


def _snippet(text, n=220):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


class LiveBM25:
    def __init__(self, path):
        t0 = time.perf_counter()
        with gzip.open(path, "rt", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        self.bm25 = BM25Okapi([tokenize(build_passage_text(r)) for r in self.records])
        self.build_ms = round((time.perf_counter() - t0) * 1000, 1)

    threshold = BM25_ABSTENTION_THRESHOLD

    def top(self, question, top_k=10):
        """(indices, scores, bm25_max) in retrieval.py's ranking order."""
        scores = self.bm25.get_scores(tokenize(question))
        order = [int(i) for i in np.argsort(scores)[::-1][:top_k]]
        return order, scores, (float(scores[order[0]]) if order else 0.0)

    def search(self, question, top_k=10):
        t0 = time.perf_counter()
        order, scores, bm25_max = self.top(question, top_k)
        abstained = bm25_max < BM25_ABSTENTION_THRESHOLD
        hits = []
        for i in order:
            r = self.records[i]
            aic = r["id"].startswith("aic-")
            hits.append({
                "id": r["id"], "institution": r.get("institution"),
                "accession_number": r.get("accession_number"),
                "title": r.get("title"), "artist": r.get("artist"), "year": r.get("year_raw"),
                "medium": r.get("medium"), "classification": r.get("classification"),
                "snippet": _snippet(r.get("description")),
                "attribution": AIC_ATTRIBUTION if aic and r.get("description") else None,
                "bm25_score": round(float(scores[i]), 3),
            })
        return {
            "question": question,
            "mode": MODE,
            "records_searched": len(self.records),
            "gate": {
                "bm25_max": round(bm25_max, 2),
                "threshold": BM25_ABSTENTION_THRESHOLD,
                "abstained": abstained,
                "reason": (f"Abstain: the best BM25 score ({bm25_max:.2f}) is below the "
                           f"{BM25_ABSTENTION_THRESHOLD} floor, so the pipeline would stop "
                           f"here, before any model call." if abstained else
                           f"Passes the BM25 floor ({bm25_max:.2f} >= "
                           f"{BM25_ABSTENTION_THRESHOLD}). The full pipeline would also "
                           f"check dense similarity, then make one model call."),
            },
            "candidates": hits,
            "note": NOTE,
            "searched_in_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


def default_path(repo_root):
    return os.getenv("CORPUS_PATH") or os.path.join(repo_root, "samples", "corpus.jsonl.gz")
