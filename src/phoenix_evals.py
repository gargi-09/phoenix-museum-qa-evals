"""
phoenix_evals.py -- The pipeline's deterministic verifiers, registered as
Phoenix evaluators, plus a standard LLM-as-a-judge baseline to compare against.

Every evaluator has the same signature:

    evaluator(payload: dict) -> EvalResult(label, score, explanation)

where `payload` is the JSON stored on each root span under
tracing.EVAL_PAYLOAD_ATTR -- question, pipeline result, the fused BM25+dense
candidate IDs, the context blocks the model was shown, and the prompt-token
audit. So every verdict is recomputed FROM THE TRACE, not copied out of the
pipeline's own result: the four reasoning.check_* functions are re-executed
here against the context the trace recorded.

The results are written back to Phoenix as span annotations (annotator_kind
CODE for deterministic evaluators, LLM for the live judge), so they appear in
the Phoenix UI's Annotations column next to the trace they judge.

WHY A BASELINE JUDGE IS INCLUDED AT ALL: the claim this project makes is that
a reference-free LLM judge, which sees only the question and the answer,
structurally cannot catch an answer that cites nothing -- "No violations
found" is fluent, on-topic and confident. The baseline exists to make that
claim checkable side by side rather than asserted.
"""
import json
import os
import re
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS, FAIL = "PASS", "FAIL"
NOT_APPLICABLE, INCONCLUSIVE = "NOT_APPLICABLE", "INCONCLUSIVE"


@dataclass
class EvalResult:
    label: str
    score: float | None
    explanation: str

    def to_dict(self):
        return asdict(self)


def _pass(explanation):
    return EvalResult(PASS, 1.0, explanation)


def _fail(explanation):
    return EvalResult(FAIL, 0.0, explanation)


# ---------------------------------------------------------------------------
# Payload accessors
# ---------------------------------------------------------------------------

def _result(payload):
    return payload.get("result") or {}


def _cited(payload):
    return list(_result(payload).get("record_ids_used") or [])


def _answer_text(payload):
    """The text a reader would see as the answer. For a response that failed
    to parse (e.g. the dropped-system-prompt case) that is the raw content --
    the pipeline refused to score it, but a downstream consumer reading the
    model output would still see it as an answer."""
    r = _result(payload)
    return (r.get("answer") or r.get("raw_content")
            or (payload.get("llm") or {}).get("raw_content") or "")


def _parsed_response(payload):
    """Reconstructs the fields of the parsed model response the checks read.
    None when the pipeline could not parse one -- which the checks themselves
    already report as 'no response to check'."""
    r = _result(payload)
    if r.get("abstained") or r.get("error"):
        return None
    return {"answer": r.get("answer"), "answerable": r.get("answerable"),
            "record_ids_used": r.get("record_ids_used") or []}


# "Everything is fine" phrasings. An assertion of ABSENCE is still a claim,
# and needs sources exactly as much as an assertion of presence does.
ALL_CLEAR_PATTERNS = re.compile(
    r"\b(no (violations?|issues?|problems?|discrepanc(y|ies)|contradictions?|errors?)"
    r"( (were |was )?(found|detected|reported))?|all (checks|fields|records) "
    r"(pass(ed)?|are (complete|consistent|verified))|fully verified)\b",
    re.IGNORECASE,
)


def _asserts_claim(payload):
    """Does the output present itself as an answer? True for answerable=True,
    for an all-clear statement, and for unparseable prose (which carries no
    answerable flag at all, so nothing marks it as a decline)."""
    r = _result(payload)
    if r.get("abstained"):
        return False
    if r.get("answerable") is True:
        return True
    text = _answer_text(payload)
    if ALL_CLEAR_PATTERNS.search(text):
        return True
    return bool(r.get("error")) and bool(text.strip())


# ---------------------------------------------------------------------------
# Evaluator 1 -- Zero-Source False Pass
# ---------------------------------------------------------------------------

def zero_source_false_pass(payload):
    """FAILS an output that asserts an answer -- including "no violations" --
    while citing zero valid source records.

    This is the failure the pipeline's own history documents: a response
    citing nothing passed 3/3 checks and scored 0.88/"high", because every
    check iterates the citations and there were none. Vacuous truth is not
    evidence. Here the absence of sources is the finding.
    """
    r = _result(payload)
    if r.get("abstained"):
        return EvalResult(NOT_APPLICABLE, None,
                          "Abstained before generation -- no claim was made.")

    candidates = set(payload.get("candidate_ids") or [])
    cited = _cited(payload)
    valid = [c for c in cited if c in candidates]

    if _asserts_claim(payload) and not valid:
        what = ("an all-clear (\"no violations\")"
                if ALL_CLEAR_PATTERNS.search(_answer_text(payload)) else "an answer")
        if r.get("error"):
            what += f" in unparseable output ({r['error'].split(':')[0]})"
        return _fail(f"FALSE PASS: output asserts {what} but cites 0 valid source "
                     f"records (cited={cited}, {len(candidates)} candidates "
                     f"available). Confidence must be 0.0.")
    if not cited:
        return _pass("Declined (answerable=false) without asserting a claim -- "
                     "zero citations is the correct shape for a decline.")
    return _pass(f"{len(valid)} valid source record(s) cited: {valid}")


# ---------------------------------------------------------------------------
# Evaluator 2 -- Citation Groundedness
# ---------------------------------------------------------------------------

def citation_groundedness(payload):
    """Every cited record ID must exist in the BM25 + dense RRF candidate set
    recorded on the retrieval span. Pure set membership; score is the fraction
    of citations that are grounded.

    Stricter than "exists in the corpus": a real record the model was never
    shown for THIS question is still an ungrounded citation.
    """
    r = _result(payload)
    if r.get("abstained"):
        return EvalResult(NOT_APPLICABLE, None, "Abstained -- nothing was cited.")

    candidates = set(payload.get("candidate_ids") or [])
    cited = _cited(payload)
    if not cited:
        if _asserts_claim(payload):
            return _fail("Answer asserts a claim with 0 citations -- nothing grounds it.")
        return EvalResult(INCONCLUSIVE, None,
                          "No citations and no claim asserted -- groundedness "
                          "cannot be measured, and is NOT reported as a pass.")

    grounded = [c for c in cited if c in candidates]
    ungrounded = [c for c in cited if c not in candidates]
    score = round(len(grounded) / len(cited), 3)
    if ungrounded:
        return EvalResult(FAIL, score,
                          f"{len(ungrounded)}/{len(cited)} citation(s) are NOT in the "
                          f"fused candidate set: {ungrounded}. Grounded: {grounded}.")
    return EvalResult(PASS, score,
                      f"All {len(cited)} citation(s) found in the "
                      f"{len(candidates)}-record RRF candidate set.")


# ---------------------------------------------------------------------------
# Evaluator 3 -- Confidence Capping
# ---------------------------------------------------------------------------

def _failed_checks(payload):
    checks = _result(payload).get("verification_checks") or {}
    return [name for name, c in checks.items()
            if not c.get("inconclusive") and not c.get("passed")]


def confidence_capping(payload):
    """When verification fails, the emitted confidence must be capped:
      - zero valid citations, or a suspected dropped system prompt
            -> no score at all (effective 0.0)
      - any scorable deterministic check failed
            -> label must not be "high"
    FAILS if a confidence escaped its cap. This audits compute_confidence()'s
    LABEL CAP and confidence_for_response()'s not_applicable short-circuit
    from the outside, so a regression in either shows up in Phoenix.
    """
    r = _result(payload)
    if r.get("abstained"):
        return EvalResult(NOT_APPLICABLE, None,
                          "Abstained before generation -- no confidence emitted.")

    conf = r.get("confidence")
    candidates = set(payload.get("candidate_ids") or [])
    valid = [c for c in _cited(payload) if c in candidates]
    dropped = bool((payload.get("prompt_audit") or {}).get("system_prompt_dropped_suspected"))
    failed = _failed_checks(payload)

    score = conf.get("score") if conf else None
    label = conf.get("label") if conf else "none"
    effective = 0.0 if score is None else score

    reasons = []
    if not valid:
        reasons.append("0 valid citations")
    if dropped:
        reasons.append("system prompt suspected dropped")
    if failed:
        reasons.append(f"failed checks {failed}")

    if not reasons:
        return _pass(f"No cap required; confidence {score} ({label}).")

    must_zero = (not valid) or dropped
    if must_zero and score is not None:
        return _fail(f"Confidence {score} ({label}) was emitted despite "
                     f"{'; '.join(reasons)} -- should be capped to 0.0.")
    if failed and label == "high":
        return _fail(f"Label 'high' despite {'; '.join(reasons)} -- the label cap "
                     f"did not fire.")

    if conf is None:
        how = f"no confidence emitted ({r.get('error', 'unscored')})"
    elif score is None:
        how = f"score withheld (label={label})"
    else:
        components = conf.get("components") or {}
        how = (f"label capped to '{label}' (score {score})"
               if components.get("label_capped_by_failed_check")
               else f"label '{label}' already below 'high' (score {score})")
    return _pass(f"Capped correctly: {'; '.join(reasons)} -> {how}; "
                 f"effective confidence {effective:.2f}.")


# ---------------------------------------------------------------------------
# Prompt integrity -- the dropped-system-prompt audit, as an evaluator
# ---------------------------------------------------------------------------

def prompt_integrity(payload):
    """FAILS when the billed prompt_tokens say the instruction block never
    reached the model (see tracing.audit_prompt_tokens for the rule)."""
    if _result(payload).get("abstained"):
        return EvalResult(NOT_APPLICABLE, None, "Abstained -- no prompt was sent.")
    audit = payload.get("prompt_audit") or {}
    status = audit.get("status")
    if status == "dropped_suspected":
        return _fail("SYSTEM PROMPT DROPPED: " + audit.get("detail", ""))
    if status == "ok":
        return _pass(audit.get("detail", "prompt_tokens consistent with instructions"))
    return EvalResult(INCONCLUSIVE, None,
                      audit.get("detail", "no prompt_tokens reported -- cannot audit"))


# ---------------------------------------------------------------------------
# The pipeline's own four deterministic verifiers, re-run from the trace
# ---------------------------------------------------------------------------

def _from_check(check):
    if check.get("inconclusive"):
        return EvalResult(INCONCLUSIVE, None, check.get("detail", ""))
    return (_pass if check.get("passed") else _fail)(check.get("detail", ""))


def _make_check_evaluator(check_name):
    def evaluator(payload):
        import reasoning
        r = _result(payload)
        if r.get("abstained"):
            return EvalResult(NOT_APPLICABLE, None, "Abstained -- nothing to verify.")
        blocks = payload.get("context_blocks") or []
        parsed = _parsed_response(payload)
        if check_name == "not_truncated":
            llm = payload.get("llm") or {}
            return _from_check(reasoning.check_not_truncated(llm or None))
        fn = getattr(reasoning, f"check_{check_name}")
        return _from_check(fn(parsed, blocks))
    evaluator.__name__ = f"verifier_{check_name}"
    evaluator.__doc__ = f"reasoning.check_{check_name}, re-executed on the traced context."
    return evaluator


DETERMINISTIC_EVALUATORS = {
    "zero_source_false_pass": zero_source_false_pass,
    "citation_groundedness": citation_groundedness,
    "confidence_capping": confidence_capping,
    "prompt_integrity": prompt_integrity,
    "verifier.citation_membership": _make_check_evaluator("citation_membership"),
    "verifier.not_recorded_violation": _make_check_evaluator("not_recorded_violation"),
    "verifier.factual_match": _make_check_evaluator("factual_match"),
    "verifier.not_truncated": _make_check_evaluator("not_truncated"),
}

# The verdict shown in the side-by-side: an output fails if ANY of these do.
HEADLINE_EVALUATORS = ("zero_source_false_pass", "citation_groundedness",
                       "confidence_capping", "prompt_integrity")


def deterministic_verdict(results):
    """Rolls the headline evaluators up into one PASS/FAIL for the comparison."""
    failures = [n for n in HEADLINE_EVALUATORS
                if n in results and results[n].label == FAIL]
    if failures:
        return _fail("; ".join(f"{n}: {results[n].explanation}" for n in failures))
    if all(results[n].label == NOT_APPLICABLE for n in HEADLINE_EVALUATORS if n in results):
        return EvalResult(NOT_APPLICABLE, None, "Nothing was generated, so nothing to verify.")
    return _pass("All headline deterministic evaluators passed or were not applicable.")


# ---------------------------------------------------------------------------
# Baseline: a standard, reference-free LLM-as-a-judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are an expert evaluator for a question-answering system over museum records.

Given a QUESTION and the system's ANSWER, decide whether the answer is relevant, coherent, and appears factually correct and complete.

QUESTION: {question}
ANSWER: {answer}

Respond ONLY with JSON: {{"label": "PASS" or "FAIL", "explanation": "one sentence"}}"""


def _judge_text(payload):
    r = _result(payload)
    return _answer_text(payload) or r.get("reason") or ""


def llm_judge_surrogate(payload):
    """OFFLINE stand-in for a reference-free LLM judge. Deterministic so the
    demo is reproducible without credentials, and given EXACTLY the judge's
    information set -- question + answer text, no candidates, no citations,
    no token counts -- because that blindness is the property under test.

    Approximates the rubric above: PASS when the answer is non-empty, reads as
    a complete sentence and is on-topic (shares a content word with the
    question). Use --judge live in the demo to run the real model instead.
    """
    if _result(payload).get("abstained"):
        return EvalResult(NOT_APPLICABLE, None, "No answer was generated to judge.")
    question = payload.get("question") or ""
    text = _judge_text(payload).strip()
    if not text:
        return _fail("Empty answer.")
    q_terms = {t for t in re.findall(r"[a-z]{4,}", question.lower())}
    a_terms = {t for t in re.findall(r"[a-z]{4,}", text.lower())}
    on_topic = bool(q_terms & a_terms)
    complete = len(text.split()) >= 6 and text.rstrip()[-1:] in ".!?\"')"
    if on_topic and complete:
        if ALL_CLEAR_PATTERNS.search(text):
            return _pass("PASS -- No violations reported; answer is on-topic and confident.")
        return _pass("PASS -- Answer is relevant, coherent and appears correct.")
    return _fail("Answer is off-topic or incomplete.")


def make_live_llm_judge(dev_mode=True):
    """A real LLM-as-a-judge through the same endpoint the pipeline uses.
    Costs one call per evaluated span; dev_mode defaults ON so it draws on the
    free sandbox budget, not the submission budget."""
    from api_client import call_haiku

    def llm_judge_live(payload):
        if _result(payload).get("abstained"):
            return EvalResult(NOT_APPLICABLE, None, "No answer was generated to judge.")
        prompt = JUDGE_PROMPT.format(question=payload.get("question", ""),
                                     answer=_judge_text(payload))
        res = call_haiku([{"role": "user", "content": prompt}], max_tokens=200,
                         dev_mode=dev_mode, stage="llm_judge", expect_json=True)
        if res.get("error_type"):
            return EvalResult(INCONCLUSIVE, None, f"judge call failed: {res['error_type']}")
        content = res.get("content") or ""
        try:
            data = json.loads(content[content.find("{"):content.rfind("}") + 1])
            label = str(data.get("label", "")).upper()
            return EvalResult(label if label in (PASS, FAIL) else INCONCLUSIVE,
                              1.0 if label == PASS else 0.0 if label == FAIL else None,
                              str(data.get("explanation", "")))
        except (ValueError, AttributeError):
            return EvalResult(INCONCLUSIVE, None, f"unparseable judge output: {content[:200]}")

    return llm_judge_live


# ---------------------------------------------------------------------------
# Running evaluators and writing them back to Phoenix
# ---------------------------------------------------------------------------

def run_evaluators(payload, evaluators):
    """{name: EvalResult}. An evaluator that raises is reported, not fatal:
    one broken evaluator must not hide the verdicts of the others."""
    out = {}
    for name, fn in evaluators.items():
        try:
            out[name] = fn(payload)
        except Exception as e:  # noqa: BLE001 -- reported as a result below
            out[name] = EvalResult(INCONCLUSIVE, None,
                                   f"evaluator raised {type(e).__name__}: {e}")
    return out


def _span_attribute(row, dotted_key):
    """Reads an attribute from a spans-dataframe row. Phoenix un-flattens
    dotted attribute keys, so 'museum_qa.eval_payload' arrives as column
    'attributes.museum_qa' holding {'eval_payload': ...}. Tries the longest
    flat column first and walks into the dict for the rest, so it works
    whichever way a given Phoenix version splits the key."""
    parts = ["attributes"] + dotted_key.split(".")
    for i in range(len(parts), 1, -1):
        col = ".".join(parts[:i])
        if col in row.index:
            value = row[col]
            for p in parts[i:]:
                value = value.get(p) if isinstance(value, dict) else None
            return value
    return None


def fetch_root_span_payloads(project_name, url, span_name=None):
    """Pulls root spans for the project back OUT of Phoenix and returns
    [(span_id, payload)]. This is what makes the evaluation run 'against the
    execution traces' rather than against in-memory results.

    Returns None if the Phoenix client is unavailable or the query fails, so
    the caller can fall back to in-memory payloads and SAY that it did.
    """
    from tracing import EVAL_PAYLOAD_ATTR, ROOT_SPAN_NAME
    span_name = span_name or ROOT_SPAN_NAME
    try:
        from phoenix.client import Client
        df = Client(base_url=url).spans.get_spans_dataframe(
            project_identifier=project_name, limit=1000)
    except Exception as e:  # noqa: BLE001
        print(f"[phoenix_evals] Could not query spans from Phoenix ({type(e).__name__}: {e})")
        return None
    if df is None or df.empty:
        return []

    if "name" in df.columns:
        df = df[df["name"] == span_name]
    id_col = "context.span_id"
    rows = []
    for idx, row in df.iterrows():
        raw = _span_attribute(row, EVAL_PAYLOAD_ATTR)
        if raw is None or (isinstance(raw, float) and raw != raw):
            continue
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            continue
        span_id = row.get(id_col) if id_col in df.columns else idx
        rows.append((span_id, payload))
    return rows


def log_annotations(rows, url):
    """rows: [{span_id, name, label, score, explanation, annotator_kind}].
    Writes them to Phoenix as span annotations, one call per annotation name.
    Returns the number of annotations written; raises nothing -- a logging
    failure is printed and the CLI report still stands on its own."""
    if not rows:
        return 0
    try:
        from phoenix.client import Client
    except ImportError as e:
        print(f"[phoenix_evals] Phoenix client unavailable, annotations not logged: {e}")
        return 0

    # Plain SpanAnnotationData dicts, NOT log_span_annotations_dataframe: the
    # dataframe path turns a None score (every INCONCLUSIVE / NOT_APPLICABLE
    # verdict) back into NaN, which then fails JSON serialisation for the
    # whole batch. Omitting the key is how the API spells "no score".
    annotations = []
    for r in rows:
        result = {k: r[k] for k in ("label", "score", "explanation")
                  if r.get(k) is not None and r.get(k) != ""}
        annotations.append({"name": r["name"], "span_id": r["span_id"],
                            "annotator_kind": r["annotator_kind"], "result": result})
    try:
        # sync=True so the count is what Phoenix inserted, not what was sent.
        inserted = Client(base_url=url).spans.log_span_annotations(
            span_annotations=annotations, sync=True)
    except Exception as e:  # noqa: BLE001
        print(f"[phoenix_evals] Failed to log annotations: {type(e).__name__}: {e}")
        return 0
    return len(inserted or [])
