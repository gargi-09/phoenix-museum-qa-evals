# Museum QA live demo API -- Cloud Run image (sample-queries-only).
#
# Contains: the FastAPI app, the pipeline modules the evaluators import, the
# recorded samples, the corpus (samples/corpus.jsonl.gz), the harness and the
# release gate. The Anthropic SDK is installed for POST /live/answer only; the
# key is never in the image (Cloud Run mounts it from Secret Manager).
# Does NOT contain: the recorders, any .env, the embedding model, torch.
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
COPY harness/ ./harness/
COPY scripts/release_gate.py ./scripts/release_gate.py
COPY samples/recorded_outputs.json samples/corpus.jsonl.gz samples/release_gate.json ./samples/

USER app
EXPOSE 8080

# exec form via sh so ${PORT} (set by Cloud Run) is expanded and uvicorn is
# PID 1 for clean shutdown. --no-access-log: Cloud Run already logs requests;
# the app logs nothing about request bodies.
CMD ["sh", "-c", "exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT} --no-access-log --workers 1"]
