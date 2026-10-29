"""Live answers for the public demo: a visitor's own question, answered by a
real Claude Haiku call and verified by the 8 deterministic evaluators.

OFF unless LIVE_ANSWERS=on. When off, nothing here is constructed: no key is
read, the Anthropic SDK is never imported, and no socket is opened (the
replay endpoints' guard test covers that).

Safeguards, all fail-closed:
  - a hard DAILY cap across every instance, kept in a Cloud Storage object
    updated with a generation precondition (atomic: two instances cannot
    both take the last slot). If the counter is unreachable, no call is made.
  - a per-visitor cap (per instance, in memory): LIVE_PER_IP_PER_HOUR.
  - questions the BM25 gate abstains on are refused before any paid call
    and cost no quota.
  - a 300-character question limit (the request model) and a 40 s client
    timeout with one SDK retry, inside Cloud Run's 60 s request timeout.
  - the key comes only from Secret Manager (ANTHROPIC_API_KEY, mounted by
    Cloud Run); nothing logs the question or the answer.

The pipeline is the real one, composed in the order reasoning.answer_question
runs it: retrieve -> assemble_context_blocks -> build_prompt -> one model
call -> parse_haiku_response -> run_verification_suite ->
confidence_for_response, plus the prompt-token audit on the exact billed
count. Differences, reported in every response:
  - retrieval is BM25 only (no dense half, no RRF): the image has no
    embedding model. So the gate is the BM25 floor alone, and confidence's
    retrieval component is 0.
  - one attempt: a failed verification abstains instead of retrying, to
    keep the paid budget predictable.
"""
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime, timezone

# Published first-party prices for claude-haiku-5-5, per million tokens (prompts
# up to 100K tokens). Used only to show an estimate next to the real usage.
PRICE_IN_PER_M, PRICE_OUT_PER_M = 0.10, 0.50

MODE_NOTE = ("Live: one real claude-haiku-5-5 call on the records BM25 retrieved, "
             "then the 8 deterministic checks. Retrieval here is BM25 only (the dense "
             "half needs an embedding model the public service does not ship), so "
             "confidence has no retrieval component. One attempt: a failed check "
             "abstains rather than paying for a retry.")


# ---------------------------------------------------------------------------
# Quota
# ---------------------------------------------------------------------------

class QuotaExhausted(Exception):
    pass


class QuotaUnavailable(Exception):
    pass


class ModelUnavailable(Exception):
    """The model cannot answer right now. reason: "credit" (the API credit
    has run out), "config" (bad or revoked key, no access) or "upstream"
    (rate limit, overload, network)."""

    def __init__(self, reason, detail=""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


PAUSE_SECONDS = 15 * 60
MESSAGES = {
    "credit": "Live answers are paused: this demo's API credit has run out.",
    "config": "Live answers are paused: the demo's API key is not working right now.",
    "upstream": "The model is busy or unreachable right now. Try again in a minute.",
}


def classify_model_error(error_type, detail):
    """Maps api_client's error_type (+ the server's own text) to a reason."""
    text = (detail or "").lower()
    if "credit balance" in text or "billing" in text or "insufficient" in text:
        return "credit"
    if error_type in ("bad_key", "permission_denied", "model_not_found", "missing_anthropic_sdk"):
        return "config"
    return "upstream"


class MemoryDailyCounter:
    """For tests and local development only: resets when the process does."""

    durable = False

    def __init__(self, cap):
        self.cap, self.day, self.used, self.lock = cap, None, 0, threading.Lock()

    def _today(self):
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def used_today(self):
        with self.lock:
            return self.used if self.day == self._today() else 0

    def reserve(self):
        with self.lock:
            if self.day != self._today():
                self.day, self.used = self._today(), 0
            if self.used >= self.cap:
                raise QuotaExhausted()
            self.used += 1
            return self.used


class GcsDailyCounter:
    """One object per UTC day in a private bucket, incremented with
    ifGenerationMatch so concurrent instances never over-spend. Uses the
    instance's service-account token from the metadata server; urllib only."""

    durable = True
    META = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"

    def __init__(self, bucket, cap, timeout=5):
        self.bucket, self.cap, self.timeout = bucket, cap, timeout
        self._token, self._token_exp = None, 0.0

    def _auth(self):
        if not self._token or time.time() > self._token_exp - 60:
            req = urllib.request.Request(self.META, headers={"Metadata-Flavor": "Google"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                tok = json.load(r)
            self._token, self._token_exp = tok["access_token"], time.time() + tok["expires_in"]
        return {"Authorization": f"Bearer {self._token}"}

    def _name(self):
        return f"live-answers/{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"

    def _read(self):
        url = (f"https://storage.googleapis.com/storage/v1/b/{self.bucket}/o/"
               f"{urllib.parse.quote(self._name(), safe='')}?alt=media")
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=self._auth()),
                                        timeout=self.timeout) as r:
                return json.load(r).get("used", 0), int(r.headers["x-goog-generation"])
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 0, 0                       # generation 0 = "must not exist yet"
            raise

    def used_today(self):
        try:
            return self._read()[0]
        except Exception as e:  # noqa: BLE001
            raise QuotaUnavailable(str(e)) from e

    def reserve(self):
        try:
            for _ in range(6):
                used, gen = self._read()
                if used >= self.cap:
                    raise QuotaExhausted()
                url = (f"https://storage.googleapis.com/upload/storage/v1/b/{self.bucket}/o"
                       f"?uploadType=media&name={urllib.parse.quote(self._name(), safe='')}"
                       f"&ifGenerationMatch={gen}")
                body = json.dumps({"used": used + 1}).encode()
                req = urllib.request.Request(url, data=body, method="POST",
                                             headers={**self._auth(), "Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=self.timeout):
                        return used + 1
                except urllib.error.HTTPError as e:
                    if e.code == 412:             # another instance won the race: re-read
                        continue
                    raise
            raise QuotaUnavailable("counter contention")
        except QuotaExhausted:
            raise
        except Exception as e:  # noqa: BLE001 -- fail closed: no counter, no call
            raise QuotaUnavailable(str(e)) from e


class PerVisitor:
    def __init__(self, per_hour):
        self.per_hour, self.hits, self.lock = per_hour, {}, threading.Lock()

    def remaining(self, key, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            q = self.hits.setdefault(key, deque())
            while q and now - q[0] >= 3600:
                q.popleft()
            return max(0, self.per_hour - len(q))

    def take(self, key, now=None):
        now = time.monotonic() if now is None else now
        if self.remaining(key, now) <= 0:
            return False
        with self.lock:
            self.hits[key].append(now)
        return True


# ---------------------------------------------------------------------------
# The live pipeline
# ---------------------------------------------------------------------------

def _ms(t0):
    return round((time.perf_counter() - t0) * 1000, 1)


class LiveAnswerer:
    def __init__(self, live_bm25, counter, per_visitor, client=None, top_k=10):
        self.live, self.counter, self.per_visitor, self.top_k = live_bm25, counter, per_visitor, top_k
        self._client = client          # injected in tests; built lazily in production
        # Per instance: after a credit or key error, stop calling the model for
        # PAUSE_SECONDS so every click does not hit the API again.
        self.paused_until, self.paused_reason = 0.0, None

    def paused(self):
        return self.paused_reason if time.time() < self.paused_until else None

    def _pause(self, reason):
        self.paused_until, self.paused_reason = time.time() + PAUSE_SECONDS, reason

    def client(self):
        if self._client is None:
            import anthropic          # only here: never imported while live answers are off
            self._client = anthropic.Anthropic(timeout=40.0, max_retries=1)
        return self._client

    def status(self, visitor):
        try:
            used = self.counter.used_today()
            available = True
        except QuotaUnavailable:
            used, available = None, False
        paused = self.paused()
        return {"enabled": True, "available": available and not paused,
                "paused_reason": paused, "message": MESSAGES.get(paused) if paused else None,
                "daily_cap": self.counter.cap,
                "used_today": used,
                "remaining_today": None if used is None else max(0, self.counter.cap - used),
                "per_visitor_per_hour": self.per_visitor.per_hour,
                "visitor_remaining": self.per_visitor.remaining(visitor),
                "model": self._model()}

    @staticmethod
    def _model():
        import api_client
        return api_client.ANTHROPIC_MODEL

    def answer(self, question, visitor):
        import api_client
        import reasoning
        from phoenix_evals import JUDGE_PROMPT
        from rescore import evaluate
        from tracing import audit_prompt_tokens

        trace, t_all = [], time.perf_counter()

        def step(name, t0, detail="", **extra):
            trace.append({"step": name, "ms": _ms(t0), "detail": detail, **extra})

        # 1. retrieve (live BM25) + the BM25 abstention floor
        t0 = time.perf_counter()
        idx, scores, bm25_max = self.live.top(question, self.top_k)
        candidates = [{"id": self.live.records[i]["id"], "record": self.live.records[i],
                       "rrf_score": float(scores[i]), "dense_similarity": None} for i in idx]
        floor = self.live.threshold
        step("retrieve", t0, f"BM25 over {len(self.live.records):,} records, top {len(candidates)}")
        t0 = time.perf_counter()
        if bm25_max < floor:
            step("gate", t0, f"abstain: best BM25 {bm25_max:.2f} < {floor}")
            return self._out(question, trace, t_all, abstained=True,
                             reason=(f"Abstained before any paid call: best BM25 score "
                                     f"{bm25_max:.2f} is below the {floor} floor."),
                             bm25_max=bm25_max, candidates=candidates)
        step("gate", t0, f"pass: best BM25 {bm25_max:.2f} >= {floor}")

        paused = self.paused()
        if paused:                     # known-broken: do not spend quota or call the API
            raise ModelUnavailable(paused)

        # 2. quota: per visitor, then the durable daily cap (fails closed)
        t0 = time.perf_counter()
        if not self.per_visitor.take(visitor):
            raise QuotaExhausted("visitor")
        used = self.counter.reserve()
        step("quota", t0, f"{used}/{self.counter.cap} live answers used today")

        # 3. context + prompt, exactly as answer_question builds them
        t0 = time.perf_counter()
        blocks, _folded = reasoning.assemble_context_blocks(candidates)
        system_prompt, messages = reasoning.build_prompt(question, blocks)
        step("build_prompt", t0, f"{len(blocks)} context blocks")

        # 4. prompt-token audit inputs: exact counts with and without the
        #    instructions (count_tokens is free)
        t0 = time.perf_counter()
        client = self.client()
        api_client._anthropic_client = client      # the transport below uses this client
        model = api_client.ANTHROPIC_MODEL
        with_i = client.messages.count_tokens(model=model, system=system_prompt,
                                              messages=messages).input_tokens
        without_i = client.messages.count_tokens(model=model, messages=messages).input_tokens
        step("count_tokens", t0, f"expected {with_i} with instructions, {without_i} without")

        # 5. the one model call, through the pipeline's Anthropic transport
        t0 = time.perf_counter()
        res = api_client._call_anthropic(messages, reasoning.ANSWER_MAX_TOKENS, "live_answer",
                                         True, system_prompt)
        usage = res.get("usage") or {}
        step("model_call", t0, model, prompt_tokens=usage.get("prompt_tokens"),
             completion_tokens=usage.get("completion_tokens"))
        if res.get("error_type"):
            reason = classify_model_error(res["error_type"], res.get("error_detail"))
            if reason in ("credit", "config"):
                self._pause(reason)
            raise ModelUnavailable(reason, res["error_type"])

        # 6. verify: parse, the 4 checks, confidence, audit on the billed count,
        #    then all 8 evaluators on the root-span-shaped payload
        t0 = time.perf_counter()
        raw = res.get("content") or ""
        llm = {"error_type": None, "is_truncated": res.get("is_truncated", False),
               "truncation_reasons": res.get("truncation_reasons") or [],
               "prior_truncation": res.get("prior_truncation"),
               "truncation_recovered": bool(res.get("truncation_recovered")), "raw_content": raw}
        parsed, perr = reasoning.parse_haiku_response(raw)
        if perr and not parsed:
            result = {"question": question, "abstained": False,
                      "error": f"failed_to_parse_response: {perr}", "answer": None, "raw_content": raw}
        else:
            checks, passed, total = reasoning.run_verification_suite(parsed, blocks, haiku_result=res)
            used_ids = set(parsed.get("record_ids_used") or [])
            contradiction = any(b["internal_contradiction"] is not None for b in blocks if b["id"] in used_ids)
            result = {"question": question, "abstained": False, "answer": parsed.get("answer"),
                      "answerable": parsed.get("answerable"),
                      "record_ids_used": parsed.get("record_ids_used"),
                      "limitations": parsed.get("limitations"),
                      "confidence": reasoning.confidence_for_response(parsed, 0.0, passed, total, contradiction),
                      "verification_checks": checks, "is_truncated": llm["is_truncated"]}
        billed = usage.get("prompt_tokens")
        payload = {
            "question": question, "result": result,
            "candidate_ids": [c["id"] for c in candidates],
            "context_blocks": [{"id": b["id"], "fields": b.get("fields"), "years": b.get("years"),
                                "internal_contradiction": bool(b.get("internal_contradiction"))}
                               for b in blocks],
            "prompt_audit": audit_prompt_tokens(billed, with_i, without_i),
            "prompt_tokens": billed, "llm": llm,
        }
        ev = evaluate(payload)
        step("verify", t0, f"deterministic verdict {ev['deterministic_verdict']['label']}")

        # 7. the real reference-free LLM judge: question + answer only
        t0 = time.perf_counter()
        judge_text = result.get("answer") or raw
        jres = api_client._call_anthropic(
            [{"role": "user", "content": JUDGE_PROMPT.format(question=question, answer=judge_text)}],
            200, "live_judge", True, None)
        judge = _parse_judge(jres)
        jusage = jres.get("usage") or {}
        step("llm_judge", t0, f"{model}, question + answer only", prompt_tokens=jusage.get("prompt_tokens"),
             completion_tokens=jusage.get("completion_tokens"))

        accepted = ev["deterministic_verdict"]["label"] == "PASS"
        step("decide", time.perf_counter(), "accepted" if accepted else "abstained: verification failed")
        return self._out(question, trace, t_all, result=result, raw=raw, evaluations=ev,
                         llm_judge=judge, accepted=accepted, payload=payload, candidates=candidates,
                         usage=usage, judge_usage=jusage, quota_used=used, bm25_max=bm25_max,
                         blocks=blocks)

    def _out(self, question, trace, t_all, candidates=(), usage=None, judge_usage=None, **kw):
        titles = {c["id"]: c["record"].get("title") for c in candidates}
        tin = (usage or {}).get("prompt_tokens") or 0
        tout = (usage or {}).get("completion_tokens") or 0
        jin = (judge_usage or {}).get("prompt_tokens") or 0
        jout = (judge_usage or {}).get("completion_tokens") or 0
        out = {
            "question": question, "mode": "live", "note": MODE_NOTE,
            "candidates": [{"id": c["id"], "title": titles[c["id"]],
                            "institution": c["record"].get("institution"),
                            "bm25_score": round(c["rrf_score"], 3)} for c in candidates],
            "trace": trace, "total_ms": _ms(t_all),
            "usage": {"answer": usage, "judge": judge_usage,
                      "estimated_cost_usd": round((tin + jin) * PRICE_IN_PER_M / 1e6
                                                  + (tout + jout) * PRICE_OUT_PER_M / 1e6, 6),
                      "pricing": f"${PRICE_IN_PER_M}/M in, ${PRICE_OUT_PER_M}/M out (claude-haiku-5-5)"},
        }
        if kw.get("abstained"):
            out.update(outcome="abstained", abstain_reason="retrieval_gate", reason=kw["reason"],
                       bm25_max=round(kw["bm25_max"], 2), model_called=False)
            return out
        if kw.get("error"):
            out.update(outcome="error", error=kw["error"], model_called=True, quota_used=kw.get("quota_used"))
            return out
        result, ev = kw["result"], kw["evaluations"]
        out.update(
            model_called=True, quota_used=kw["quota_used"], bm25_max=round(kw["bm25_max"], 2),
            outcome="accepted" if kw["accepted"] else "abstained",
            abstain_reason=None if kw["accepted"] else "verification_failed",
            output={"raw": kw["raw"], "answer": result.get("answer"), "answerable": result.get("answerable"),
                    "record_ids_used": result.get("record_ids_used"), "error": result.get("error"),
                    "limitations": result.get("limitations")},
            confidence=result.get("confidence"),
            prompt_audit=kw["payload"]["prompt_audit"],
            evaluations={k: v for k, v in ev.items() if k not in ("deterministic_verdict", "llm_judge_surrogate")},
            deterministic_verdict=ev["deterministic_verdict"],
            llm_judge_surrogate=ev["llm_judge_surrogate"],
            llm_judge=kw["llm_judge"],
            context_titles={b["id"]: (b.get("fields") or {}).get("title") for b in kw["blocks"]},
        )
        return out


def _parse_judge(res):
    """The same parsing phoenix_evals.make_live_llm_judge applies."""
    if res.get("error_type"):
        return {"label": "INCONCLUSIVE", "score": None, "explanation": f"judge call failed: {res['error_type']}"}
    content = res.get("content") or ""
    try:
        data = json.loads(content[content.find("{"):content.rfind("}") + 1])
        label = str(data.get("label", "")).upper()
        return {"label": label if label in ("PASS", "FAIL") else "INCONCLUSIVE",
                "score": 1.0 if label == "PASS" else 0.0 if label == "FAIL" else None,
                "explanation": str(data.get("explanation", ""))}
    except (ValueError, AttributeError):
        return {"label": "INCONCLUSIVE", "score": None, "explanation": f"unparseable judge output: {content[:200]}"}


def from_env(live_bm25):
    """None unless LIVE_ANSWERS=on AND a durable counter is configured (or the
    explicit local-only memory counter). Fails closed on anything missing."""
    if os.getenv("LIVE_ANSWERS", "").strip().lower() != "on":
        return None
    cap = int(os.getenv("LIVE_DAILY_CAP", "50"))
    per_ip = int(os.getenv("LIVE_PER_IP_PER_HOUR", "5"))
    bucket = os.getenv("COUNTER_BUCKET", "").strip()
    if bucket:
        counter = GcsDailyCounter(bucket, cap)
    elif os.getenv("LIVE_ALLOW_MEMORY_COUNTER") == "1":
        counter = MemoryDailyCounter(cap)
    else:
        return None
    return LiveAnswerer(live_bm25, counter, PerVisitor(per_ip))
