"""
server/app.py -- The live demo API (FastAPI, deployed on Cloud Run).

SAMPLE-QUERIES-ONLY. This service never calls a model:
  - LLM_PROVIDER=disabled is set below, BEFORE any pipeline import, so
    api_client never loads .env, never imports an HTTP client, never reads
    ANTHROPIC_API_KEY, and refuses every model call.
  - It serves the fixed samples in samples/recorded_outputs.json ("recorded
    model outputs", written offline by record_samples.py) selected BY ID.
    Free-text questions are refused.
  - What runs live, per request: the post-generation pipeline (parse,
    verification checks, confidence, prompt-token audit) and all 8
    deterministic evaluators plus the judge baseline, on either the recorded
    answer or a visitor's edited one (src/rescore.py, which composes the
    pipeline's own functions). Retrieval is replayed from the recording.

Endpoints:
  GET  /health                 liveness; no I/O
  GET  /samples                the sample list
  GET  /samples/{sample_id}    a recorded sample, evaluated live
  POST /evaluate               {"sample_id", "answer"?, "system_prompt_delivered"?}

Run locally:
  uvicorn server.app:app --port 8080
  SAMPLES_PATH=docs/data/sandbox.json uvicorn server.app:app --port 8080   # fixture data
"""
import os

# Must precede every pipeline import. Forced, not defaulted: an operator
# setting LLM_PROVIDER=anthropic on the service must not re-enable calls.
os.environ["LLM_PROVIDER"] = "disabled"

import json
import sys
import time
from collections import deque
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from rescore import evaluate, rescore  # noqa: E402  (after LLM_PROVIDER is set)

SAMPLES_ONLY = "Live demo runs on recorded samples only."
DEFAULT_SAMPLES_PATH = os.path.join(REPO_ROOT, "samples", "recorded_outputs.json")

ALLOWED_ORIGINS = ["https://gargi-09.github.io"]
# Local development only: any port on localhost / 127.0.0.1. No wildcard.
LOCALHOST_ORIGIN_REGEX = r"^http://(localhost|127\.0\.0\.1)(:\d{1,5})?$"

MAX_BODY_BYTES = 16 * 1024
MAX_ANSWER_CHARS = 8000
EVALUATOR_NAMES = ("zero_source_false_pass", "citation_groundedness", "confidence_capping",
                   "prompt_integrity", "verifier.citation_membership",
                   "verifier.not_recorded_violation", "verifier.factual_match",
                   "verifier.not_truncated")


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

class SampleStore:
    """Loads the recorded file once. The sandbox fixture file
    (docs/data/sandbox.json) has the same per-sample shape, so it can stand
    in for local development; its label says what it is."""

    def __init__(self, path):
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        self.label = doc.get("label") or doc.get("source") or "unlabelled sample file"
        self.notice = doc.get("notice")
        self.samples = {}
        for s in doc.get("scenarios") or []:
            if "id" not in s or "payload" not in s:
                raise ValueError(f"{path}: every sample needs 'id' and 'payload'")
            self.samples[s["id"]] = s

    def summary(self, s):
        result = s["payload"].get("result") or {}
        return {"id": s["id"], "kind": s.get("kind") or s.get("scenario"),
                "expect": s.get("expect") or s.get("why"),
                "question": s["payload"]["question"],
                "abstained": bool(result.get("abstained")),
                "seed": s.get("seed")}


def _view(store, s, payload, source):
    """One response shape for a recorded or an edited answer."""
    t0 = time.perf_counter()
    ev = evaluate(payload)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    result = payload.get("result") or {}
    titles = {b["id"]: (b.get("fields") or {}).get("title") for b in payload.get("context_blocks") or []}
    llm = payload.get("llm") or {}
    return {
        "sample_id": s["id"],
        "kind": s.get("kind") or s.get("scenario"),
        "question": payload["question"],
        "source": source,
        "data_label": store.label,
        "notice": ("Model output is a recorded sample" if source == "recorded"
                   else "Model output is a visitor edit of a recorded sample")
                  + "; the evaluators ran live on this request. No model was called.",
        "output": {
            "raw": llm.get("raw_content"),
            "answer": result.get("answer"),
            "answerable": result.get("answerable"),
            "record_ids_used": result.get("record_ids_used"),
            "error": result.get("error"),
        },
        "retrieval": {
            "abstained": bool(result.get("abstained")),
            "reason": result.get("reason"),
            "candidates": [{"id": c, "title": titles.get(c)} for c in payload.get("candidate_ids") or []],
        },
        "pipeline": {
            "confidence": result.get("confidence"),
            "prompt_tokens": payload.get("prompt_tokens"),
            "prompt_audit": payload.get("prompt_audit"),
            "verification_checks": result.get("verification_checks"),
        },
        "evaluations": {k: ev[k] for k in EVALUATOR_NAMES if k in ev},
        "deterministic_verdict": ev.get("deterministic_verdict"),
        "llm_judge_surrogate": ev.get("llm_judge_surrogate"),
        "recorded": s.get("recorded"),
        "seed": s.get("seed"),
        "evaluated_in_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Safety middleware
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window limit per client IP, in memory (per instance; with max 2
    instances the effective ceiling is at most 2x). Cloud Run puts the client
    address first in X-Forwarded-For."""

    def __init__(self, per_minute):
        self.per_minute = per_minute
        self.hits = {}
        self.lock = Lock()

    def allow(self, key, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            q = self.hits.setdefault(key, deque())
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False, max(1, int(60 - (now - q[0])) + 1)
            q.append(now)
            if len(self.hits) > 10000:          # bound memory: drop idle clients
                for k in [k for k, v in self.hits.items() if not v or now - v[-1] >= 60]:
                    del self.hits[k]
            return True, 0


def client_ip(scope):
    for name, value in scope.get("headers") or []:
        if name == b"x-forwarded-for":
            return value.decode("latin-1").split(",")[0].strip()
    client = scope.get("client")
    return client[0] if client else "unknown"


class SafetyMiddleware:
    """Pure ASGI: body-size cap (declared and streamed) and per-IP rate limit.
    Logs nothing about requests."""

    def __init__(self, app, limiter, max_body=MAX_BODY_BYTES):
        self.app, self.limiter, self.max_body = app, limiter, max_body

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        if scope["method"] != "OPTIONS" and scope["path"] != "/health":
            ok, retry = self.limiter.allow(client_ip(scope))
            if not ok:
                return await JSONResponse(
                    {"detail": f"Rate limit: {self.limiter.per_minute} requests per minute. "
                               f"Try again in {retry}s."},
                    status_code=429, headers={"Retry-After": str(retry)})(scope, receive, send)

        for name, value in scope.get("headers") or []:
            if name == b"content-length" and value.isdigit() and int(value) > self.max_body:
                return await JSONResponse({"detail": f"Request body over {self.max_body} bytes."},
                                          status_code=413)(scope, receive, send)

        seen = 0

        async def limited_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_body:
                    raise _BodyTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await JSONResponse({"detail": f"Request body over {self.max_body} bytes."},
                               status_code=413)(scope, receive, send)


class _BodyTooLarge(Exception):
    pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    # extra="forbid": a free-text field ("question", "query", "prompt") is
    # rejected with the samples-only message rather than silently ignored.
    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1, max_length=32)
    answer: Optional[str] = Field(default=None, max_length=MAX_ANSWER_CHARS)
    system_prompt_delivered: Optional[bool] = None


def create_app(samples_path=None, rate_limit_per_minute=None):
    store = SampleStore(samples_path or os.getenv("SAMPLES_PATH") or DEFAULT_SAMPLES_PATH)
    limiter = RateLimiter(rate_limit_per_minute
                          or int(os.getenv("RATE_LIMIT_PER_MINUTE", "30")))

    app = FastAPI(title="Museum QA live demo API (recorded samples)",
                  version="1.0", docs_url="/docs", redoc_url=None)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        extra = [e for e in exc.errors() if e.get("type") == "extra_forbidden"]
        detail = (SAMPLES_ONLY + " Send a sample_id from GET /samples." if extra
                  else "Invalid request: expected {sample_id, answer?, system_prompt_delivered?}.")
        return JSONResponse({"detail": detail}, status_code=422 if not extra else 400)

    @app.exception_handler(404)
    async def not_found(request: Request, exc):
        msg = getattr(exc, "detail", None)
        if not isinstance(msg, str) or msg == "Not Found":
            msg = f"Not found. {SAMPLES_ONLY} See GET /samples."
        return JSONResponse({"detail": msg}, status_code=404)

    @app.get("/health")
    def health():
        return {"status": "ok", "samples": len(store.samples), "data_label": store.label,
                "model_calls": "disabled"}

    @app.get("/samples")
    def list_samples():
        return {"data_label": store.label, "notice": store.notice,
                "samples": [store.summary(s) for s in store.samples.values()]}

    def get_sample(sample_id):
        s = store.samples.get(sample_id)
        if s is None:
            raise HTTPException(404, f"Unknown sample_id {sample_id!r}. {SAMPLES_ONLY} "
                                     f"See GET /samples.")
        return s

    @app.get("/samples/{sample_id}")
    def sample(sample_id: str):
        s = get_sample(sample_id)
        return _view(store, s, s["payload"], "recorded")

    @app.post("/evaluate")
    def evaluate_sample(req: EvaluateRequest):
        s = get_sample(req.sample_id)
        recorded_raw = (s["payload"].get("llm") or {}).get("raw_content")
        delivered_default = not s.get("dropped_by_script")
        delivered = delivered_default if req.system_prompt_delivered is None else req.system_prompt_delivered
        unchanged = (req.answer is None or req.answer == recorded_raw) and delivered == delivered_default
        if unchanged or (s["payload"].get("result") or {}).get("abstained"):
            return _view(store, s, s["payload"], "recorded")
        payload = rescore(s, req.answer if req.answer is not None else (recorded_raw or ""), delivered)
        return _view(store, s, payload, "edited")

    @app.api_route("/ask", methods=["GET", "POST"])
    @app.api_route("/query", methods=["GET", "POST"])
    @app.api_route("/answer", methods=["GET", "POST"])
    def free_text():
        return JSONResponse({"detail": SAMPLES_ONLY + " Pick a sample_id from GET /samples."},
                            status_code=400)

    # Order matters: the last middleware added is the outermost. CORS wraps
    # the safety layer so 429/413 responses still carry CORS headers.
    app.add_middleware(SafetyMiddleware, limiter=limiter)
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                       allow_origin_regex=LOCALHOST_ORIGIN_REGEX,
                       allow_methods=["GET", "POST", "OPTIONS"],
                       allow_headers=["Content-Type"], allow_credentials=False, max_age=600)
    app.state.store = store
    app.state.limiter = limiter

    # Warm-up: the first evaluation imports the pipeline modules (measured
    # ~350 ms locally). Paying that at startup keeps it out of the first
    # visitor's request. Pure computation on a recorded payload; no I/O.
    if store.samples:
        evaluate(next(iter(store.samples.values()))["payload"])
    return app


app = create_app()
