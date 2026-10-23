"""
tracing.py -- OpenInference / OpenTelemetry instrumentation for the pipeline,
exported to a local Arize Phoenix session.

WHY MANUAL SPANS AND NOT AN AUTO-INSTRUMENTOR: api_client.call_haiku() posts
to the endpoint with raw `requests`. There is no LangChain, LlamaIndex, OpenAI
or Anthropic SDK in the call path, so none of the openinference-instrumentation-*
packages has anything to hook. Spans are therefore emitted by wrapping the
pipeline's own seams, using OpenInference semantic conventions so Phoenix
renders them as first-class RETRIEVER / LLM / GUARDRAIL / EVALUATOR spans.

WHY WRAPPING AND NOT EDITING reasoning.py: answer_question() resolves
call_haiku, build_prompt and run_verification_suite through reasoning's module
namespace at call time, so replacing those names is enough to trace every
stage. The pipeline's logic is untouched, and an uninstrumented run behaves
exactly as before -- tracing is opt-in via instrument_pipeline().

One question produces one trace:

    museum_qa.answer_question          CHAIN      root; carries the eval payload
      retrieval.hybrid_rrf             RETRIEVER  BM25 + bge-small, fused by RRF
      prompt.token_audit               GUARDRAIL  expected prompt_tokens, pre-call
      llm.generate                     LLM        reported prompt_tokens vs expected
      verifier.deterministic_suite     EVALUATOR  the four deterministic checks

Usage:
    from tracing import start_phoenix, instrument_pipeline
    session = start_phoenix(project_name="museum-qa")
    instrument_pipeline()
"""
import contextvars
import functools
import json
import os
import socket
import sys
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:6006")
DEFAULT_PROJECT = os.getenv("PHOENIX_PROJECT_NAME", "museum-qa-deterministic-verifiers")

# Attribute namespace for everything that is not an OpenInference convention.
NS = "museum_qa"

# Root-span attribute holding everything the Phoenix evaluators need to re-run
# the deterministic checks from the trace alone (see phoenix_evals.py).
EVAL_PAYLOAD_ATTR = f"{NS}.eval_payload"

ROOT_SPAN_NAME = "museum_qa.answer_question"

_tracer = None
_instrumented = False

# Per-question scratch state shared between the wrapped stages. A ContextVar
# rather than a global so concurrent questions (threads, asyncio) cannot read
# each other's audit numbers.
_question_state = contextvars.ContextVar("museum_qa_question_state", default=None)

# span_id -> the exact eval payload JSON written onto that root span. Lets the
# demo fall back to evaluating in-process if querying Phoenix fails, while
# still evaluating byte-for-byte what the trace carries.
_recorded_payloads = {}


def get_recorded_payload(span_id):
    raw = _recorded_payloads.get(span_id)
    return json.loads(raw) if raw else None


# ---------------------------------------------------------------------------
# Phoenix session + tracer provider
# ---------------------------------------------------------------------------

def phoenix_is_running(url=PHOENIX_URL, timeout=0.5):
    """True if something is already listening on the Phoenix port -- e.g. a
    `phoenix serve` the user started in another terminal. Launching a second
    in-process app on the same port would fail, so reuse it instead."""
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6006
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def start_phoenix(project_name=DEFAULT_PROJECT, launch=True, url=PHOENIX_URL):
    """Starts (or attaches to) Phoenix and registers an OTLP tracer provider.

    Returns a dict: {url, project_name, launched, tracer_provider}.

    launch=True calls phoenix.launch_app() only when nothing is already
    listening on the port. Raises RuntimeError with an actionable message if
    the Phoenix packages are missing -- tracing is the point of the demo, so
    silently continuing untraced would make every downstream claim false.
    """
    global _tracer

    try:
        from phoenix.otel import register
    except ImportError as e:
        raise RuntimeError(
            "Phoenix tracing packages are not installed. Run:\n"
            "    pip install -r requirements.txt\n"
            f"(import error: {e})"
        ) from e

    launched, server_process = None, None
    if launch and not phoenix_is_running(url):
        launched, server_process = _launch(url)

    if not phoenix_is_running(url, timeout=3.0):
        raise RuntimeError(
            f"No Phoenix collector is reachable at {url}. Start one with "
            f"`phoenix serve` (or re-run without --no-launch)."
        )

    # batch=False: the demo is short-lived and evaluates its own traces right
    # after producing them, so spans must be exported as each one ends rather
    # than whenever a batch fills.
    #
    # set_global_tracer_provider=False is REQUIRED when Phoenix runs in this
    # same process (launch_app). FastAPI instruments itself with the GLOBAL
    # provider, so a global registration made Phoenix's own server trace every
    # /v1/traces request it received and export that span -- synchronously,
    # from inside the request handler -- back to itself. Measured: every
    # export timed out ("Failed to export spans batch"), the span query timed
    # out, and no annotations could be written. get_tracer() below uses this
    # provider directly, so nothing needs it to be global.
    tracer_provider = register(
        project_name=project_name,
        endpoint=f"{url.rstrip('/')}/v1/traces",
        batch=False,
        set_global_tracer_provider=False,
        verbose=False,
    )
    _tracer = tracer_provider.get_tracer(__name__)
    return {"url": url, "project_name": project_name, "launched": launched,
            "server_process": server_process, "tracer_provider": tracer_provider}


def _launch(url, attempts=2, subprocess_wait_s=90):
    """Returns (how, process): how is 'launch_app' or 'subprocess'.

    phoenix.launch_app() is tried first. It gives the server thread a hard
    5-second start budget, and a COLD first start (imports plus creating and
    migrating the database) was measured exceeding it on Windows with
    "server took too long to start". The second attempt is warm and usually
    fits; if it still does not, fall back to `phoenix serve` in a child process
    and poll the port, which has no such limit.

    use_temp_dir=False keeps traces in Phoenix's working directory (~/.phoenix
    by default) so they survive the demo exiting -- and avoids a Windows
    PermissionError when the temp-dir cleanup races the still-open database.
    """
    try:
        import phoenix as px
    except ImportError as e:
        raise RuntimeError(
            "arize-phoenix is not installed, so a local Phoenix app cannot be "
            "launched. Run `pip install -r requirements.txt`, or start one with "
            f"`phoenix serve` and re-run. (import error: {e})"
        ) from e

    port = urlparse(url).port or 6006
    # launch_app(port=...) is deprecated in favour of the environment variable.
    os.environ.setdefault("PHOENIX_PORT", str(port))
    last_error = None
    for _ in range(attempts):
        try:
            px.launch_app(use_temp_dir=False)
            return "launch_app", None
        except RuntimeError as e:
            last_error = e
            if phoenix_is_running(url):
                return "launch_app", None

    import subprocess
    import time
    print(f"[tracing] phoenix.launch_app() failed ({last_error}); starting "
          f"`phoenix serve` in a child process instead...")
    env = dict(os.environ, PHOENIX_PORT=str(port))
    proc = subprocess.Popen([sys.executable, "-m", "phoenix.server.main", "serve"],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + subprocess_wait_s
    while time.time() < deadline:
        if phoenix_is_running(url):
            return "subprocess", proc
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"Phoenix did not start: launch_app() said {last_error!r}, and "
                       f"`phoenix serve` did not open port {port} within "
                       f"{subprocess_wait_s}s. Start it manually and use --no-launch.")


def get_tracer():
    """The registered tracer, or OpenTelemetry's global (a no-op tracer when
    nothing is registered) -- so instrumented code never crashes untraced."""
    if _tracer is not None:
        return _tracer
    from opentelemetry import trace
    return trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _json(value, limit=None):
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if limit and len(text) > limit:
        return text[:limit] + f"... [truncated, {len(text)} chars]"
    return text


def _set(span, key, value):
    """Sets an attribute, skipping None and coercing anything OTel can't hold."""
    if value is None:
        return
    if isinstance(value, (bool, int, float, str)):
        span.set_attribute(key, value)
    elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        span.set_attribute(key, list(value))
    else:
        span.set_attribute(key, _json(value))


def _usage_tokens(usage):
    """(prompt, completion, total) from either OpenAI- or Anthropic-shaped usage."""
    if not usage:
        return None, None, None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return prompt, completion, total


def _record_exception(span, exc):
    from opentelemetry.trace import Status, StatusCode
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


def _ok(span):
    from opentelemetry.trace import Status, StatusCode
    span.set_status(Status(StatusCode.OK))


# ---------------------------------------------------------------------------
# Prompt token audit -- the "silently dropped system prompt" detector
# ---------------------------------------------------------------------------

def audit_prompt_tokens(reported_prompt_tokens, expected_with_instructions,
                        expected_without_instructions):
    """Decides whether the billed prompt_tokens look like the instructions
    arrived. Returns a dict of audit fields.

    This is the measurement api_client.py's header documents, made automatic:
    the proxy billed a bare "hi" at 8 tokens and "hi" + a ~13-token system
    prompt ALSO at 8 -- the instructions were discarded with a 200 OK. Nothing
    errors in that case, so the only evidence is the token count.

    RULE: the report is closer to the no-instructions estimate than to the
    with-instructions estimate, i.e. reported < without + instructions/2.
    Both estimates come from estimate_tokens_preflight()'s ~4 chars/token
    heuristic, so this is a SUSPICION flag, not proof -- which is why the field
    is called `suspected`. It is robust in the case that matters, because the
    instructions block is ~450 tokens and a drop moves the count by all of it.
    """
    instruction_tokens = max(0, expected_with_instructions - expected_without_instructions)
    if reported_prompt_tokens is None:
        return {"status": "unmeasured", "system_prompt_dropped_suspected": False,
                "instruction_tokens_expected": instruction_tokens,
                "detail": "endpoint reported no prompt_tokens -- delivery cannot be audited"}

    threshold = expected_without_instructions + instruction_tokens / 2
    dropped = instruction_tokens > 0 and reported_prompt_tokens < threshold
    delta = reported_prompt_tokens - expected_with_instructions
    return {
        "status": "dropped_suspected" if dropped else "ok",
        "system_prompt_dropped_suspected": dropped,
        "instruction_tokens_expected": instruction_tokens,
        "prompt_tokens_delta": delta,
        "prompt_tokens_ratio": round(reported_prompt_tokens / expected_with_instructions, 3)
        if expected_with_instructions else None,
        "detail": (
            f"reported prompt_tokens {reported_prompt_tokens} is below the "
            f"{threshold:.0f}-token midpoint between the with-instructions estimate "
            f"({expected_with_instructions}) and the without-instructions estimate "
            f"({expected_without_instructions}) -- the ~{instruction_tokens}-token "
            f"instruction block was most likely NOT delivered"
            if dropped else
            f"reported prompt_tokens {reported_prompt_tokens} is consistent with the "
            f"instructions arriving (estimate {expected_with_instructions}, "
            f"delta {delta:+d})"
        ),
    }


# ---------------------------------------------------------------------------
# Stage wrappers
# ---------------------------------------------------------------------------

def _wrap_retrieve(retrieve):
    """RETRIEVER span around HybridRetriever.retrieve (or any object with the
    same (candidates, abstain, best_sim, signals) contract)."""
    from openinference.semconv.trace import (DocumentAttributes,
                                             OpenInferenceSpanKindValues,
                                             SpanAttributes)

    @functools.wraps(retrieve)
    def traced_retrieve(question, *args, **kwargs):
        with get_tracer().start_as_current_span("retrieval.hybrid_rrf") as span:
            _set(span, SpanAttributes.OPENINFERENCE_SPAN_KIND,
                 OpenInferenceSpanKindValues.RETRIEVER.value)
            _set(span, SpanAttributes.INPUT_VALUE, question)
            try:
                candidates, abstain, best_sim, signals = retrieve(question, *args, **kwargs)
            except Exception as e:
                _record_exception(span, e)
                raise

            prefix = SpanAttributes.RETRIEVAL_DOCUMENTS
            for i, c in enumerate(candidates):
                rec = c.get("record") or {}
                _set(span, f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_ID}", c["id"])
                _set(span, f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_SCORE}",
                     float(c.get("rrf_score") or 0.0))
                _set(span, f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_CONTENT}",
                     f"{rec.get('title')} -- {rec.get('artist')} ({rec.get('year_raw')})")
                _set(span, f"{prefix}.{i}.{DocumentAttributes.DOCUMENT_METADATA}",
                     {"institution": rec.get("institution"),
                      "dense_similarity": c.get("dense_similarity")})

            _set(span, f"{NS}.retrieval.candidate_count", len(candidates))
            _set(span, f"{NS}.retrieval.candidate_ids", [c["id"] for c in candidates])
            _set(span, f"{NS}.retrieval.abstain", bool(abstain))
            _set(span, f"{NS}.retrieval.best_dense_similarity", float(best_sim))
            _set(span, f"{NS}.retrieval.bm25_max", float(signals.get("bm25_max", 0.0)))
            if signals.get("abstain_reasons"):
                _set(span, f"{NS}.retrieval.abstain_reasons", signals["abstain_reasons"])
            _set(span, SpanAttributes.OUTPUT_VALUE,
                 _json([c["id"] for c in candidates]))

            state = _question_state.get()
            if state is not None:
                state["candidate_ids"] = [c["id"] for c in candidates]
                state["retrieval_signals"] = signals
            _ok(span)
            return candidates, abstain, best_sim, signals

    traced_retrieve._oi_traced = True
    return traced_retrieve


def _wrap_build_prompt(build_prompt):
    """GUARDRAIL span: estimates, BEFORE the paid call, how many prompt tokens
    the endpoint should bill with and without the instruction block. The LLM
    span compares the real count against these."""
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
    from api_client import MESSAGE_FORMAT, estimate_tokens_preflight

    @functools.wraps(build_prompt)
    def traced_build_prompt(question, context_blocks, *args, **kwargs):
        with get_tracer().start_as_current_span("prompt.token_audit") as span:
            _set(span, SpanAttributes.OPENINFERENCE_SPAN_KIND,
                 OpenInferenceSpanKindValues.GUARDRAIL.value)
            _set(span, SpanAttributes.INPUT_VALUE, question)
            try:
                instructions, messages = build_prompt(question, context_blocks, *args, **kwargs)
            except Exception as e:
                _record_exception(span, e)
                raise

            instruction_tokens = estimate_tokens_preflight(instructions)
            message_tokens = estimate_tokens_preflight(messages)
            expected = instruction_tokens + message_tokens

            _set(span, f"{NS}.prompt.instruction_tokens_expected", instruction_tokens)
            _set(span, f"{NS}.prompt.context_tokens_expected", message_tokens)
            _set(span, f"{NS}.prompt_tokens_expected", expected)
            _set(span, f"{NS}.prompt.context_block_count", len(context_blocks))
            _set(span, f"{NS}.prompt.context_block_ids", [b["id"] for b in context_blocks])
            _set(span, f"{NS}.prompt.message_format", MESSAGE_FORMAT)
            _set(span, SpanAttributes.OUTPUT_VALUE,
                 f"expected prompt_tokens ~{expected} "
                 f"({instruction_tokens} instructions + {message_tokens} question/records)")

            state = _question_state.get()
            if state is not None:
                state["expected_with_instructions"] = expected
                state["expected_without_instructions"] = message_tokens
                state["context_blocks"] = [
                    {"id": b["id"], "fields": b.get("fields"), "years": b.get("years"),
                     "internal_contradiction": bool(b.get("internal_contradiction"))}
                    for b in context_blocks
                ]
            _ok(span)
            return instructions, messages

    traced_build_prompt._oi_traced = True
    return traced_build_prompt


def _wrap_call_haiku(call_haiku):
    """LLM span around the single paid call. Records OpenInference token
    counts AND the explicit museum_qa.prompt_tokens audit fields."""
    from openinference.semconv.trace import (MessageAttributes,
                                             OpenInferenceSpanKindValues,
                                             SpanAttributes)
    from api_client import MESSAGE_FORMAT, MODEL_NAME

    @functools.wraps(call_haiku)
    def traced_call_haiku(messages, *args, **kwargs):
        with get_tracer().start_as_current_span("llm.generate") as span:
            _set(span, SpanAttributes.OPENINFERENCE_SPAN_KIND,
                 OpenInferenceSpanKindValues.LLM.value)
            _set(span, SpanAttributes.LLM_MODEL_NAME, MODEL_NAME)
            _set(span, SpanAttributes.LLM_INVOCATION_PARAMETERS, {
                "max_tokens": kwargs.get("max_tokens"),
                "temperature": kwargs.get("temperature"),
                "message_format": MESSAGE_FORMAT,
                "dev_mode": kwargs.get("dev_mode"),
                "stage": kwargs.get("stage"),
            })
            system = kwargs.get("system")
            in_prefix = SpanAttributes.LLM_INPUT_MESSAGES
            idx = 0
            if system:
                _set(span, f"{in_prefix}.{idx}.{MessageAttributes.MESSAGE_ROLE}", "system")
                _set(span, f"{in_prefix}.{idx}.{MessageAttributes.MESSAGE_CONTENT}", system)
                idx += 1
            for m in messages:
                _set(span, f"{in_prefix}.{idx}.{MessageAttributes.MESSAGE_ROLE}", m.get("role"))
                content = m.get("content")
                _set(span, f"{in_prefix}.{idx}.{MessageAttributes.MESSAGE_CONTENT}",
                     content if isinstance(content, str) else _json(content))
                idx += 1
            _set(span, SpanAttributes.INPUT_VALUE, _json(messages, limit=20000))
            _set(span, SpanAttributes.INPUT_MIME_TYPE, "application/json")

            try:
                result = call_haiku(messages, *args, **kwargs)
            except Exception as e:
                _record_exception(span, e)
                raise

            prompt_t, completion_t, total_t = _usage_tokens(result.get("usage"))
            _set(span, SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_t)
            _set(span, SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_t)
            _set(span, SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_t)
            # Explicit, un-namespaced-by-OpenInference copy, as requested: this
            # is the number the system-prompt-drop audit is built on.
            _set(span, f"{NS}.prompt_tokens", prompt_t)

            state = _question_state.get() or {}
            audit = audit_prompt_tokens(
                prompt_t,
                state.get("expected_with_instructions", 0),
                state.get("expected_without_instructions", 0),
            )
            for k, v in audit.items():
                _set(span, f"{NS}.prompt_audit.{k}", v)
            if _question_state.get() is not None:
                state["prompt_audit"] = audit
                state["prompt_tokens"] = prompt_t
                state["llm"] = {
                    "error_type": result.get("error_type"),
                    "is_truncated": result.get("is_truncated"),
                    "truncation_reasons": result.get("truncation_reasons"),
                    "prior_truncation": result.get("prior_truncation"),
                    "truncation_recovered": result.get("truncation_recovered"),
                    "raw_content": result.get("content"),
                }

            _set(span, f"{NS}.llm.error_type", result.get("error_type"))
            _set(span, f"{NS}.llm.is_truncated", bool(result.get("is_truncated")))
            _set(span, f"{NS}.llm.call_seq", result.get("call_seq"))
            _set(span, f"{NS}.llm.budget_remaining", result.get("budget_remaining"))

            content = result.get("content")
            if content is not None:
                out_prefix = SpanAttributes.LLM_OUTPUT_MESSAGES
                _set(span, f"{out_prefix}.0.{MessageAttributes.MESSAGE_ROLE}", "assistant")
                _set(span, f"{out_prefix}.0.{MessageAttributes.MESSAGE_CONTENT}", content)
                _set(span, SpanAttributes.OUTPUT_VALUE, content)

            if result.get("error_type"):
                from opentelemetry.trace import Status, StatusCode
                span.set_status(Status(StatusCode.ERROR, result["error_type"]))
            else:
                _ok(span)
            return result

    traced_call_haiku._oi_traced = True
    return traced_call_haiku


def _wrap_verification(run_verification_suite):
    """EVALUATOR span around the deterministic verification suite: the hook
    where the pipeline judges its own answer before anyone else does."""
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

    @functools.wraps(run_verification_suite)
    def traced_verification(parsed_response, context_blocks, *args, **kwargs):
        with get_tracer().start_as_current_span("verifier.deterministic_suite") as span:
            _set(span, SpanAttributes.OPENINFERENCE_SPAN_KIND,
                 OpenInferenceSpanKindValues.EVALUATOR.value)
            _set(span, SpanAttributes.INPUT_VALUE, _json(parsed_response, limit=8000))
            try:
                checks, passed, total = run_verification_suite(
                    parsed_response, context_blocks, *args, **kwargs)
            except Exception as e:
                _record_exception(span, e)
                raise

            valid_ids = {b["id"] for b in context_blocks}
            cited = list((parsed_response or {}).get("record_ids_used") or [])
            _set(span, f"{NS}.citations.count", len(cited))
            _set(span, f"{NS}.citations.valid_count", sum(1 for c in cited if c in valid_ids))
            _set(span, f"{NS}.citations.fabricated_count",
                 sum(1 for c in cited if c not in valid_ids))
            _set(span, f"{NS}.verification.passed", passed)
            _set(span, f"{NS}.verification.scorable", total)
            for name, check in checks.items():
                verdict = ("inconclusive" if check.get("inconclusive")
                           else "pass" if check.get("passed") else "fail")
                _set(span, f"{NS}.check.{name}", verdict)
                _set(span, f"{NS}.check.{name}.detail", check.get("detail"))
            _set(span, SpanAttributes.OUTPUT_VALUE, f"{passed}/{total} scorable checks passed")
            _ok(span)
            return checks, passed, total

    traced_verification._oi_traced = True
    return traced_verification


def _wrap_answer_question(answer_question):
    """Root CHAIN span. Also instruments the retriever it is handed, and
    stores the eval payload the Phoenix evaluators re-run checks from."""
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

    @functools.wraps(answer_question)
    def traced_answer_question(question, retriever, *args, **kwargs):
        if not getattr(retriever.retrieve, "_oi_traced", False):
            retriever.retrieve = _wrap_retrieve(retriever.retrieve)

        state = {}
        token = _question_state.set(state)
        try:
            with get_tracer().start_as_current_span(ROOT_SPAN_NAME) as span:
                _set(span, SpanAttributes.OPENINFERENCE_SPAN_KIND,
                     OpenInferenceSpanKindValues.CHAIN.value)
                _set(span, SpanAttributes.INPUT_VALUE, question)
                try:
                    result = answer_question(question, retriever, *args, **kwargs)
                except Exception as e:
                    _record_exception(span, e)
                    raise

                cited = list(result.get("record_ids_used") or [])
                context_ids = {b["id"] for b in state.get("context_blocks", [])}
                confidence = result.get("confidence") or {}
                audit = state.get("prompt_audit") or {}

                _set(span, f"{NS}.outcome",
                     "abstained" if result.get("abstained")
                     else "error" if result.get("error") else "answered")
                _set(span, f"{NS}.citations.count", len(cited))
                _set(span, f"{NS}.citations.valid_count",
                     sum(1 for c in cited if c in context_ids))
                _set(span, f"{NS}.answerable", result.get("answerable"))
                _set(span, f"{NS}.confidence.score", confidence.get("score"))
                _set(span, f"{NS}.confidence.label", confidence.get("label"))
                _set(span, f"{NS}.prompt_tokens", state.get("prompt_tokens"))
                _set(span, f"{NS}.prompt_audit.status", audit.get("status"))
                _set(span, f"{NS}.system_prompt_dropped_suspected",
                     bool(audit.get("system_prompt_dropped_suspected")))
                if result.get("error"):
                    _set(span, f"{NS}.error", result["error"])

                payload_json = _json({
                    "question": question,
                    "result": result,
                    "candidate_ids": state.get("candidate_ids", []),
                    "context_blocks": state.get("context_blocks", []),
                    "prompt_audit": audit,
                    "prompt_tokens": state.get("prompt_tokens"),
                    "llm": state.get("llm") or {},
                })
                _set(span, EVAL_PAYLOAD_ATTR, payload_json)
                # For an unparseable response, show what the model actually
                # said: that text is what a downstream reader -- or a judge --
                # would see, and it is the evidence for the failure.
                _set(span, SpanAttributes.OUTPUT_VALUE,
                     result.get("answer") or result.get("raw_content")
                     or result.get("reason") or result.get("error") or "")
                if result.get("error"):
                    from opentelemetry.trace import Status, StatusCode
                    span.set_status(Status(StatusCode.ERROR, result["error"]))
                else:
                    _ok(span)

                # Handed back so the caller can attach evaluations to exactly
                # this trace without a round-trip query.
                span_id = format(span.get_span_context().span_id, "016x")
                _recorded_payloads[span_id] = payload_json
                result["_trace"] = {
                    "span_id": span_id,
                    "trace_id": format(span.get_span_context().trace_id, "032x"),
                }
                return result
        finally:
            _question_state.reset(token)

    traced_answer_question._oi_traced = True
    return traced_answer_question


def instrument_pipeline():
    """Wraps reasoning's stage functions with OpenInference spans. Idempotent.

    Call AFTER any test double has been installed on reasoning.call_haiku (as
    the fixture demo does), so the double is what gets traced.
    """
    global _instrumented
    import reasoning

    for name, wrapper in (("answer_question", _wrap_answer_question),
                          ("build_prompt", _wrap_build_prompt),
                          ("call_haiku", _wrap_call_haiku),
                          ("run_verification_suite", _wrap_verification)):
        fn = getattr(reasoning, name)
        if not getattr(fn, "_oi_traced", False):
            setattr(reasoning, name, wrapper(fn))
    _instrumented = True
    return reasoning.answer_question
