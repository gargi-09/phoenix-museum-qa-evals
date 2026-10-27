# Museum QA live demo API -- Cloud Run image (sample-queries-only).
#
# Contains: the FastAPI app, the pipeline modules the evaluators import, the
# recorded samples, and the slim corpus (samples/corpus.jsonl.gz) the live
# BM25 endpoint indexes. Does NOT contain: record_samples.py, any .env, the
# embedding model or embeddings, torch, the Anthropic SDK, or an HTTP client.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LLM_PROVIDER=disabled \
    PORT=8080

WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt \
 && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

# Only the modules on the serving import path (server -> rescore -> reasoning,
# phoenix_evals, tracing, api_client and what reasoning imports).
COPY src/rescore.py src/reasoning.py src/retrieval.py src/entity_resolution.py \
     src/contradiction.py src/ingest.py src/memory.py src/api_client.py \
     src/phoenix_evals.py src/tracing.py ./src/
COPY server/ ./server/
COPY samples/recorded_outputs.json samples/corpus.jsonl.gz ./samples/

USER app
EXPOSE 8080

# exec form via sh so ${PORT} (set by Cloud Run) is expanded and uvicorn is
# PID 1 for clean shutdown. --no-access-log: Cloud Run already logs requests;
# the app logs nothing about request bodies.
CMD ["sh", "-c", "exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT} --no-access-log --workers 1"]
