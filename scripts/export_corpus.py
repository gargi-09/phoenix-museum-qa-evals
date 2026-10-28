"""
export_corpus.py -- Writes the slim corpus the live retrieval endpoint serves:

    python scripts/export_corpus.py      # data/normalized.jsonl -> samples/corpus.jsonl.gz

Ships every normalized field: the live-answer path builds the same context
blocks the pipeline does (years, dimensions, "[not recorded]" markers, the
fields entity resolution and contradiction checks read), so its verifiers
see exactly what the pipeline's verifiers see. Checks that every passage
rebuilt from the file is byte-identical to the pipeline's, so live BM25
scores are the pipeline's scores.

Licences: Cleveland Museum of Art metadata is CC0 1.0. Art Institute of
Chicago metadata is CC0 1.0 except `description`, which is CC BY 4.0
(c) the Art Institute of Chicago. The API returns that attribution with every
AIC description it serves.
"""
import gzip
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from retrieval import build_passage_text, load_records  # noqa: E402

SRC = os.path.join(REPO_ROOT, "data", "normalized.jsonl")
OUT = os.path.join(REPO_ROOT, "samples", "corpus.jsonl.gz")


def main():
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} is missing. Run: python src/main.py fetch && python src/main.py ingest")
    records = load_records(SRC)
    slim = [dict(r) for r in records]
    bad = [r["id"] for r, s in zip(records, slim) if build_passage_text(r) != build_passage_text(s)]
    if bad:
        sys.exit(f"passage text changed for {len(bad)} records, e.g. {bad[:3]}")
    # mtime=0: the same corpus always produces the same bytes.
    with gzip.GzipFile(OUT, "wb", mtime=0) as f:
        for s in slim:
            f.write((json.dumps(s, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
    print(f"[corpus] {len(slim)} records -> {os.path.relpath(OUT, REPO_ROOT)} "
          f"({os.path.getsize(OUT) / 1e6:.2f} MB); passages identical to the pipeline's")


if __name__ == "__main__":
    main()
