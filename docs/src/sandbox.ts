/*
 * sandbox.ts -- the dashboard's Evaluator Sandbox engine, in strict TypeScript.
 * Compiled by `npm run build:sandbox` to docs/sandbox.js, which is committed
 * (GitHub Pages serves docs/ with no build step) and checked in CI to match
 * this source.
 *
 * A line-by-line port of the code that runs AFTER the model call:
 *   src/reasoning.py   parse_haiku_response, check_citation_membership,
 *                      check_not_recorded_violation, check_factual_match,
 *                      check_not_truncated, score_checks, compute_confidence,
 *                      confidence_for_response, answer_question's result shape
 *   src/api_client.py  split_code_fence, detect_truncation
 *   src/ingest.py      normalize_year
 *   src/tracing.py     audit_prompt_tokens
 *   src/phoenix_evals.py  every evaluator, deterministic_verdict,
 *                      llm_judge_surrogate
 *
 * Retrieval is NOT re-run: each scenario carries the candidate set and context
 * blocks recorded on a real trace (docs/data/sandbox.json, written by
 * src/export_sandbox_data.py). So editing the model's output re-scores it
 * against the exact records the pipeline retrieved -- with no model and no key.
 *
 * Parity is checked, not assumed: selfTest() re-scores every exported
 * scenario and compares each verdict with the Python run's. The page shows the
 * result; `node docs/sandbox.js docs/data/sandbox.json` prints it.
 *
 * Types describe the data as it arrives. Fields the model writes (answer,
 * answerable, record_ids_used, ...) are `unknown`: model output is untrusted
 * JSON, and the logic below coerces it exactly as the Python does.
 */

type Label = 'PASS' | 'FAIL' | 'NOT_APPLICABLE' | 'INCONCLUSIVE';

interface EvalResult { label: Label; score: number | null; explanation: string; }

interface Check { passed: boolean; inconclusive?: boolean; recovered?: boolean; detail: string; }

interface ContextBlock {
  id: string;
  fields?: Record<string, string>;
  years?: number[];
  internal_contradiction?: boolean;
}

/** The model's JSON answer, as parsed. Every field is model-written. */
interface ParsedResponse {
  [key: string]: unknown;
  answer?: unknown;
  answerable?: unknown;
  record_ids_used?: unknown;
  limitations?: unknown;
}

interface LlmResult {
  error_type: string | null;
  is_truncated: boolean;
  truncation_reasons?: string[];
  prior_truncation?: unknown;
  truncation_recovered?: boolean;
  raw_content?: string;
}

interface ConfidenceComponents {
  retrieval_score: number;
  verification_score: number;
  contradiction_penalty: number;
  label_capped_by_failed_check: boolean;
}

interface Confidence {
  score: number | null;
  label: string;
  components: ConfidenceComponents | null;
  reason?: string;
}

interface PromptAudit {
  status: string;
  system_prompt_dropped_suspected: boolean;
  instruction_tokens_expected?: number;
  prompt_tokens_delta?: number;
  detail: string;
}

interface Result {
  question?: string;
  abstained?: boolean;
  error?: string;
  answer?: unknown;
  raw_content?: string;
  answerable?: unknown;
  record_ids_used?: unknown;
  limitations?: unknown;
  confidence?: Confidence | null;
  verification_checks?: Record<string, Check>;
  is_truncated?: boolean;
  reason?: string;
}

interface Payload {
  question: string;
  result?: Result;
  candidate_ids?: string[];
  context_blocks?: ContextBlock[];
  prompt_audit?: PromptAudit;
  prompt_tokens?: number | null;
  llm?: LlmResult | null;
}

interface AuditInputs { expected_with_instructions: number; expected_without_instructions: number; }

interface Scenario {
  id: string;
  payload: Payload;
  audit_inputs?: AuditInputs | null;
  max_tokens?: number;
  best_similarity?: number;
  dropped_by_script?: boolean;
  python_verdicts: Record<string, { label: string; score: number | null }>;
}

interface SandboxData { scenarios?: Scenario[]; }

interface Mismatch { scenario: string; evaluator: string; python: string; js: string; }

interface SelfTestResult { total: number; matched: number; mismatches: Mismatch[]; }

type Evaluator = (p: Payload) => EvalResult;
type CheckFn = (parsed: ParsedResponse | null, blocks: ContextBlock[]) => Check;

interface SandboxApi {
  runPipeline: (scenario: Scenario, rawContent: string, delivered: boolean) => Payload;
  evaluate: (payload: Payload) => Record<string, EvalResult>;
  selfTest: (data: SandboxData) => SelfTestResult;
  parseResponse: (raw: string) => [ParsedResponse | null, string | null];
  normalizeYear: (s: unknown) => number[];
  HEADLINE: string[];
}

interface Root { MuseumQASandbox?: SandboxApi; }

(function (root: Root) {
  'use strict';

  var PASS: Label = 'PASS', FAIL: Label = 'FAIL', NA: Label = 'NOT_APPLICABLE', INC: Label = 'INCONCLUSIVE';

  // ---------------------------------------------------------------------
  // ingest.normalize_year
  // ---------------------------------------------------------------------
  function normalizeYear(s: unknown): number[] {
    if (!s || !String(s).trim()) return [];
    var str = String(s);
    var years: Record<number, number> = {};
    var shorthand = /\b((?:1[0-9]|20)[0-9]{2})[\/–\-](\d{2})\b/g, m: RegExpExecArray | null;
    while ((m = shorthand.exec(str))) {
      years[+m[1]] = 1;
      years[+(m[1].slice(0, 2) + m[2])] = 1;
    }
    var four = /\b(1[0-9]{3}|20[0-9]{2})\b/g;
    while ((m = four.exec(str))) years[+m[1]] = 1;
    return Object.keys(years).map(Number).sort(function (a, b) { return a - b; });
  }

  // ---------------------------------------------------------------------
  // api_client.split_code_fence / detect_truncation
  // ---------------------------------------------------------------------
  function splitCodeFence(text: string): [string, 'none' | 'closed' | 'unclosed'] {
    var stripped = text.trim();
    if (stripped.indexOf('```') !== 0) return [stripped, 'none'];
    var body = stripped.replace(/^```[A-Za-z0-9_+-]*[ \t]*\r?\n?/, '');
    var r = body.replace(/\s+$/, '');
    if (r.slice(-3) === '```') return [r.slice(0, -3).trim(), 'closed'];
    return [body.trim(), 'unclosed'];
  }

  function jsonErrorPos(text: string): number | null {
    try { JSON.parse(text); return null; } catch (e) {
      var msg = String((e as Error).message || '');
      var m = /position (\d+)/.exec(msg);
      if (m) return +m[1];
      if (/end of JSON|Unterminated/i.test(msg)) return text.length;
      return 0;
    }
  }

  function detectTruncation(text: string, usage: { completion_tokens?: number } | null,
                            maxTokens: number, expectJson: boolean): [boolean, string[]] {
    var reasons: string[] = [];
    if (!text) return [false, reasons];
    var completion = usage && usage.completion_tokens;
    if (completion && maxTokens && completion >= maxTokens * 0.98) {
      reasons.push('completion_tokens (' + completion + ') is within 2% of max_tokens_requested (' + maxTokens + ')');
    }
    var sf = splitCodeFence(text), stripped = sf[0];
    if (sf[1] === 'unclosed') reasons.push('response opens a markdown code fence but never closes it');
    if (stripped && '.!?"\'”)}]'.indexOf(stripped.slice(-1)) === -1) {
      reasons.push('response does not end in terminal punctuation (ends with: ' + JSON.stringify(stripped.slice(-20)) + ')');
    }
    if (/\b(and|or|but|the|a|an|with|to|of|in|on|for|is|was|,)\s*$/i.test(stripped)) {
      reasons.push('response ends with a dangling conjunction/article/comma');
    }
    if (expectJson) {
      var pos = jsonErrorPos(stripped);
      if (pos !== null && pos >= stripped.length - 2) {
        reasons.push('JSON parse failed at/near the end of the string (pos ' + pos + ' of ' + stripped.length + ')');
      }
    }
    return [reasons.length > 0, reasons];
  }

  // ---------------------------------------------------------------------
  // reasoning.parse_haiku_response
  // ---------------------------------------------------------------------
  var REQUIRED_KEYS = ['answer', 'answerable', 'record_ids_used', 'record_ids_excluded', 'limitations'];

  function parseResponse(raw: string): [ParsedResponse | null, string | null] {
    if (!raw) return [null, 'empty_response'];
    var text = raw.trim();
    if (text.indexOf('```') === 0) {
      text = text.replace(/^`+|`+$/g, '');
      if (text.indexOf('json') === 0) text = text.slice(4);
      text = text.trim();
    }
    var start = text.indexOf('{'), end = text.lastIndexOf('}');
    if (start === -1 || end === -1 || end < start) return [null, 'no_json_found'];
    var value: unknown;
    try { value = JSON.parse(text.slice(start, end + 1)); } catch (e) {
      return [null, 'json_decode_error: ' + (e as Error).message];
    }
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return [null, 'json_decode_error: not an object'];
    }
    var parsed = value as ParsedResponse;
    var missing = REQUIRED_KEYS.filter(function (k) { return !(k in parsed); });
    if (missing.length) return [parsed, 'missing_keys: ' + missing.join(', ')];
    return [parsed, null];
  }

  // ---------------------------------------------------------------------
  // reasoning verification checks
  // ---------------------------------------------------------------------
  var INCONCLUSIVE_NO_CITATIONS = 'INCONCLUSIVE -- the response cited no records, so this check had nothing to examine. It did not pass; it could not run.';

  function idsOf(parsed: ParsedResponse | null): string[] {
    var v = parsed && parsed.record_ids_used;
    return Array.isArray(v) ? v.map(String) : [];
  }

  function uniq(a: string[]): string[] { return a.filter(function (x, i) { return a.indexOf(x) === i; }); }

  function checkCitationMembership(parsed: ParsedResponse | null, blocks: ContextBlock[]): Check {
    if (!parsed) return { passed: false, detail: 'no response to check' };
    var valid = blocks.map(function (b) { return b.id; });
    var used = uniq(idsOf(parsed));
    if (!used.length) return { passed: true, inconclusive: true, detail: INCONCLUSIVE_NO_CITATIONS };
    var fab = used.filter(function (u) { return valid.indexOf(u) === -1; });
    return { passed: fab.length === 0, detail: fab.length ? 'fabricated citations: ' + JSON.stringify(fab) : 'all citations valid' };
  }

  function checkNotRecordedViolation(parsed: ParsedResponse | null, blocks: ContextBlock[]): Check {
    if (!parsed) return { passed: false, detail: 'no response to check' };
    var answer = String(parsed.answer || '').toLowerCase();
    var used = uniq(idsOf(parsed));
    if (!used.length) return { passed: true, inconclusive: true, detail: INCONCLUSIVE_NO_CITATIONS };
    var byId: Record<string, ContextBlock> = {};
    blocks.forEach(function (b) { byId[b.id] = b; });
    var guard = ['not recorded', 'unknown'].filter(function (t) { return answer.indexOf(t) !== -1; });
    var violations: [string, string][] = [], inScope: string[] = [];
    used.forEach(function (rid) {
      var b = byId[rid];
      if (!b) return;
      var fields = b.fields || {};
      Object.keys(fields).forEach(function (f) {
        if (fields[f] === '[not recorded]') {
          inScope.push(rid + '.' + f);
          if (answer.indexOf(f) !== -1 && !guard.length) violations.push([rid, f]);
        }
      });
    });
    if (violations.length) return { passed: false, detail: 'possible violations: ' + JSON.stringify(violations) };
    if (guard.length && inScope.length) {
      return { passed: true, inconclusive: true, detail: 'INCONCLUSIVE -- the answer contains ' + JSON.stringify(guard) + ', which suppresses this substring-based check. ' + inScope.length + ' [not recorded] field(s) went unexamined.' };
    }
    if (!inScope.length) return { passed: true, detail: 'no violations possible -- no cited record has a [not recorded] field for the answer to fill in' };
    return { passed: true, detail: 'no violations detected across ' + inScope.length + ' [not recorded] field(s) in scope: ' + JSON.stringify(inScope) };
  }

  function checkFactualMatch(parsed: ParsedResponse | null, blocks: ContextBlock[]): Check {
    if (!parsed) return { passed: false, detail: 'no response to check' };
    var used = uniq(idsOf(parsed));
    if (!used.length) return { passed: true, inconclusive: true, detail: INCONCLUSIVE_NO_CITATIONS };
    var answerYears = normalizeYear(String(parsed.answer || ''));
    if (!answerYears.length) return { passed: true, detail: 'no year mentioned in answer -- nothing to check' };
    var byId: Record<string, ContextBlock> = {};
    blocks.forEach(function (b) { byId[b.id] = b; });
    var mismatches: { record_id: string; record_years: number[]; answer_years: number[] }[] = [];
    used.forEach(function (rid) {
      var b = byId[rid];
      if (!b) return;
      var ry = b.years || [];
      if (!ry.length) return;
      if (!answerYears.some(function (y) { return ry.indexOf(y) !== -1; })) {
        mismatches.push({ record_id: rid, record_years: ry, answer_years: answerYears });
      }
    });
    return { passed: mismatches.length === 0, detail: mismatches.length ? 'year mismatches: ' + JSON.stringify(mismatches) : 'years consistent with cited records' };
  }

  function checkNotTruncated(llm: LlmResult | null): Check {
    if (!llm) return { passed: true, inconclusive: true, detail: 'INCONCLUSIVE -- no API result was available to inspect.' };
    if (llm.is_truncated) {
      return { passed: false, detail: 'TRUNCATION SUSPECTED -- signals that fired: ' + JSON.stringify(llm.truncation_reasons || []) };
    }
    if (llm.prior_truncation) return { passed: true, recovered: true, detail: 'TRUNCATION DETECTED AND RECOVERED' };
    return { passed: true, detail: 'no truncation signal fired' };
  }

  function scoreChecks(checks: Record<string, Check>): [number, number] {
    var scorable = Object.keys(checks).map(function (k) { return checks[k]; })
      .filter(function (c) { return !c.inconclusive; });
    return [scorable.filter(function (c) { return c.passed; }).length, scorable.length];
  }

  // ---------------------------------------------------------------------
  // reasoning confidence
  // ---------------------------------------------------------------------
  function round2(x: number): number { return Number(x.toFixed(2)); }

  function computeConfidence(bestSim: number, passed: number, total: number, contradiction: boolean): Confidence {
    var vs = total ? passed / total : 0;
    var rs = Math.min(bestSim, 1.0);
    var penalty = contradiction ? 0.3 : 0.0;
    var score = Math.max(0, Math.min(1, 0.5 * rs + 0.5 * vs - penalty));
    var reported = round2(score);
    var raw = reported >= 0.75 ? 'high' : reported >= 0.45 ? 'medium' : 'low';
    var anyFailed = !!total && passed < total;
    var label = anyFailed && raw === 'high' ? 'medium' : raw;
    return { score: reported, label: label, components: {
      retrieval_score: round2(rs), verification_score: round2(vs),
      contradiction_penalty: penalty, label_capped_by_failed_check: label !== raw } };
  }

  function confidenceForResponse(parsed: ParsedResponse | null, bestSim: number, passed: number,
                                 total: number, contradiction: boolean): Confidence {
    var cited = idsOf(parsed);
    var answerable = parsed ? parsed.answerable : null;
    if (cited.length && answerable) return computeConfidence(bestSim, passed, total, contradiction);
    return { score: null, label: 'not_applicable', components: null,
      reason: !cited.length ? 'no factual claim was asserted and no record was cited'
        : 'the model set answerable=false, so no factual claim was asserted' };
  }

  // ---------------------------------------------------------------------
  // tracing.audit_prompt_tokens
  // ---------------------------------------------------------------------
  function auditPromptTokens(reported: number | null | undefined, withI: number, withoutI: number): PromptAudit {
    var instr = Math.max(0, withI - withoutI);
    if (reported === null || reported === undefined) {
      return { status: 'unmeasured', system_prompt_dropped_suspected: false,
        detail: 'endpoint reported no prompt_tokens -- delivery cannot be audited' };
    }
    var threshold = withoutI + instr / 2;
    var dropped = instr > 0 && reported < threshold;
    var delta = reported - withI;
    return {
      status: dropped ? 'dropped_suspected' : 'ok',
      system_prompt_dropped_suspected: dropped,
      instruction_tokens_expected: instr, prompt_tokens_delta: delta,
      detail: dropped
        ? 'reported prompt_tokens ' + reported + ' is below the ' + Math.round(threshold) + '-token midpoint between the with-instructions estimate (' + withI + ') and the without-instructions estimate (' + withoutI + ') -- the ~' + instr + '-token instruction block was most likely NOT delivered'
        : 'reported prompt_tokens ' + reported + ' is consistent with the instructions arriving (estimate ' + withI + ', delta ' + (delta >= 0 ? '+' : '') + delta + ')'
    };
  }

  // ---------------------------------------------------------------------
  // The post-generation half of reasoning.answer_question()
  // ---------------------------------------------------------------------
  /*
   * Re-runs everything after the model call for one scenario, given the raw
   * model output and whether the instructions were delivered. Returns an
   * eval payload in the same shape tracing.py writes onto the root span.
   */
  function runPipeline(scenario: Scenario, rawContent: string, delivered: boolean): Payload {
    var base = scenario.payload;
    if (base.result && base.result.abstained) return base;   // gate fired before any call

    // Every exported scenario that reaches this point carries audit_inputs;
    // the casts keep the original's arithmetic when one does not.
    var a = (scenario.audit_inputs || {}) as AuditInputs;
    var billed = delivered ? a.expected_with_instructions : a.expected_without_instructions;
    var audit = auditPromptTokens(billed, a.expected_with_instructions, a.expected_without_instructions);

    var completion = Math.max(1, Math.floor((rawContent || '').length / 4));
    var trunc = detectTruncation(rawContent, { completion_tokens: completion }, scenario.max_tokens || 1024, true);
    var llm: LlmResult = { error_type: null, is_truncated: trunc[0], truncation_reasons: trunc[1],
      prior_truncation: null, truncation_recovered: false, raw_content: rawContent };

    var blocks = base.context_blocks || [];
    var pr = parseResponse(rawContent), parsed = pr[0], err = pr[1];
    var result: Result;
    if (err && !parsed) {
      result = { question: base.question, abstained: false,
        error: 'failed_to_parse_response: ' + err, answer: null, raw_content: rawContent };
    } else {
      // parseResponse only returns a null answer together with an error.
      var ok = parsed as ParsedResponse;
      var checks: Record<string, Check> = {
        citation_membership: checkCitationMembership(ok, blocks),
        not_recorded_violation: checkNotRecordedViolation(ok, blocks),
        factual_match: checkFactualMatch(ok, blocks),
        not_truncated: checkNotTruncated(llm)
      };
      var sc = scoreChecks(checks);
      var used = idsOf(ok);
      var contradiction = blocks.some(function (b) { return used.indexOf(b.id) !== -1 && !!b.internal_contradiction; });
      result = { question: base.question, abstained: false,
        answer: ok.answer, answerable: ok.answerable,
        record_ids_used: ok.record_ids_used, limitations: ok.limitations,
        confidence: confidenceForResponse(ok, scenario.best_similarity as number, sc[0], sc[1], contradiction),
        verification_checks: checks, is_truncated: trunc[0] };
    }
    return { question: base.question, result: result, candidate_ids: base.candidate_ids,
      context_blocks: blocks, prompt_audit: audit, prompt_tokens: billed, llm: llm };
  }

  // ---------------------------------------------------------------------
  // phoenix_evals.py
  // ---------------------------------------------------------------------
  var ALL_CLEAR = /\b(no (violations?|issues?|problems?|discrepanc(y|ies)|contradictions?|errors?)( (were |was )?(found|detected|reported))?|all (checks|fields|records) (pass(ed)?|are (complete|consistent|verified))|fully verified)\b/i;

  function R(label: Label, score: number | null, explanation: string): EvalResult {
    return { label: label, score: score, explanation: explanation };
  }
  function res(p: Payload): Result { return p.result || {}; }
  function cited(p: Payload): string[] { var v = res(p).record_ids_used; return Array.isArray(v) ? v.map(String) : []; }
  // The model's `answer` is unknown JSON; like the Python, it is used as text.
  function answerText(p: Payload): string {
    var r = res(p);
    return (r.answer || r.raw_content || (p.llm || {} as Partial<LlmResult>).raw_content || '') as string;
  }

  function assertsClaim(p: Payload): boolean {
    var r = res(p);
    if (r.abstained) return false;
    if (r.answerable === true) return true;
    var t = answerText(p);
    if (ALL_CLEAR.test(t)) return true;
    return !!r.error && !!t.trim();
  }

  function zeroSourceFalsePass(p: Payload): EvalResult {
    var r = res(p);
    if (r.abstained) return R(NA, null, 'Abstained before generation -- no claim was made.');
    var cands = p.candidate_ids || [], c = cited(p);
    var valid = c.filter(function (x) { return cands.indexOf(x) !== -1; });
    if (assertsClaim(p) && !valid.length) {
      var what = ALL_CLEAR.test(answerText(p)) ? 'an all-clear ("no violations")' : 'an answer';
      if (r.error) what += ' in unparseable output (' + r.error.split(':')[0] + ')';
      return R(FAIL, 0, 'FALSE PASS: output asserts ' + what + ' but cites 0 valid source records (cited=' + JSON.stringify(c) + ', ' + cands.length + ' candidates available). Confidence must be 0.0.');
    }
    if (!c.length) return R(PASS, 1, 'Declined (answerable=false) without asserting a claim -- zero citations is the correct shape for a decline.');
    return R(PASS, 1, valid.length + ' valid source record(s) cited: ' + JSON.stringify(valid));
  }

  function citationGroundedness(p: Payload): EvalResult {
    if (res(p).abstained) return R(NA, null, 'Abstained -- nothing was cited.');
    var cands = p.candidate_ids || [], c = cited(p);
    if (!c.length) {
      if (assertsClaim(p)) return R(FAIL, 0, 'Answer asserts a claim with 0 citations -- nothing grounds it.');
      return R(INC, null, 'No citations and no claim asserted -- groundedness cannot be measured, and is NOT reported as a pass.');
    }
    var g = c.filter(function (x) { return cands.indexOf(x) !== -1; });
    var u = c.filter(function (x) { return cands.indexOf(x) === -1; });
    var score = Math.round((g.length / c.length) * 1000) / 1000;
    if (u.length) return R(FAIL, score, u.length + '/' + c.length + ' citation(s) are NOT in the fused candidate set: ' + JSON.stringify(u) + '. Grounded: ' + JSON.stringify(g) + '.');
    return R(PASS, score, 'All ' + c.length + ' citation(s) found in the ' + cands.length + '-record RRF candidate set.');
  }

  function failedChecks(p: Payload): string[] {
    var ch = res(p).verification_checks || {};
    return Object.keys(ch).filter(function (k) { return !ch[k].inconclusive && !ch[k].passed; });
  }

  function confidenceCapping(p: Payload): EvalResult {
    var r = res(p);
    if (r.abstained) return R(NA, null, 'Abstained before generation -- no confidence emitted.');
    var conf = r.confidence, cands = p.candidate_ids || [];
    var valid = cited(p).filter(function (x) { return cands.indexOf(x) !== -1; });
    var dropped = !!(p.prompt_audit || {} as Partial<PromptAudit>).system_prompt_dropped_suspected;
    var failed = failedChecks(p);
    var score = conf ? conf.score : null, label = conf ? conf.label : 'none';
    var reasons: string[] = [];
    if (!valid.length) reasons.push('0 valid citations');
    if (dropped) reasons.push('system prompt suspected dropped');
    if (failed.length) reasons.push('failed checks ' + JSON.stringify(failed));
    if (!reasons.length) return R(PASS, 1, 'No cap required; confidence ' + score + ' (' + label + ').');
    var mustZero = !valid.length || dropped;
    if (mustZero && score !== null && score !== undefined) {
      return R(FAIL, 0, 'Confidence ' + score + ' (' + label + ') was emitted despite ' + reasons.join('; ') + ' -- should be capped to 0.0.');
    }
    if (failed.length && label === 'high') return R(FAIL, 0, "Label 'high' despite " + reasons.join('; ') + ' -- the label cap did not fire.');
    var how: string;
    if (!conf) how = 'no confidence emitted (' + (r.error || 'unscored') + ')';
    else if (score === null) how = 'score withheld (label=' + label + ')';
    else how = (conf.components || {} as Partial<ConfidenceComponents>).label_capped_by_failed_check
      ? "label capped to '" + label + "' (score " + score + ')'
      : "label '" + label + "' already below 'high' (score " + score + ')';
    return R(PASS, 1, 'Capped correctly: ' + reasons.join('; ') + ' -> ' + how + '; effective confidence ' + (score === null || score === undefined ? 0 : score).toFixed(2) + '.');
  }

  function promptIntegrity(p: Payload): EvalResult {
    if (res(p).abstained) return R(NA, null, 'Abstained -- no prompt was sent.');
    var a = p.prompt_audit || {} as Partial<PromptAudit>;
    if (a.status === 'dropped_suspected') return R(FAIL, 0, 'SYSTEM PROMPT DROPPED: ' + (a.detail || ''));
    if (a.status === 'ok') return R(PASS, 1, a.detail || 'prompt_tokens consistent with instructions');
    return R(INC, null, a.detail || 'no prompt_tokens reported -- cannot audit');
  }

  function parsedFromResult(p: Payload): ParsedResponse | null {
    var r = res(p);
    if (r.abstained || r.error) return null;
    return { answer: r.answer, answerable: r.answerable, record_ids_used: r.record_ids_used || [] };
  }

  function fromCheck(c: Check): EvalResult {
    if (c.inconclusive) return R(INC, null, c.detail || '');
    return c.passed ? R(PASS, 1, c.detail || '') : R(FAIL, 0, c.detail || '');
  }

  function checkEvaluator(fn: CheckFn): Evaluator {
    return function (p) {
      if (res(p).abstained) return R(NA, null, 'Abstained -- nothing to verify.');
      return fromCheck(fn(parsedFromResult(p), p.context_blocks || []));
    };
  }

  var EVALUATORS: Record<string, Evaluator> = {
    zero_source_false_pass: zeroSourceFalsePass,
    citation_groundedness: citationGroundedness,
    confidence_capping: confidenceCapping,
    prompt_integrity: promptIntegrity,
    'verifier.citation_membership': checkEvaluator(checkCitationMembership),
    'verifier.not_recorded_violation': checkEvaluator(checkNotRecordedViolation),
    'verifier.factual_match': checkEvaluator(checkFactualMatch),
    'verifier.not_truncated': function (p) {
      if (res(p).abstained) return R(NA, null, 'Abstained -- nothing to verify.');
      return fromCheck(checkNotTruncated(p.llm || null));
    }
  };

  var HEADLINE = ['zero_source_false_pass', 'citation_groundedness', 'confidence_capping', 'prompt_integrity'];

  // Headline four OR any verifier.* check failing fails the verdict, as in
  // phoenix_evals.deterministic_verdict. Inconclusive is never a failure.
  function deterministicVerdict(results: Record<string, EvalResult>): EvalResult {
    var names = Object.keys(results).filter(function (n) {
      return HEADLINE.indexOf(n) !== -1 || n.indexOf('verifier.') === 0;
    });
    var failures = names.filter(function (n) { return results[n].label === FAIL; });
    if (failures.length) return R(FAIL, 0, failures.map(function (n) { return n + ': ' + results[n].explanation; }).join('; '));
    if (HEADLINE.every(function (n) { return !results[n] || results[n].label === NA; })) {
      return R(NA, null, 'Nothing was generated, so nothing to verify.');
    }
    return R(PASS, 1, 'All deterministic evaluators passed, or could not run (inconclusive) or did not apply.');
  }

  function llmJudgeSurrogate(p: Payload): EvalResult {
    if (res(p).abstained) return R(NA, null, 'No answer was generated to judge.');
    var q = String(p.question || '').toLowerCase();
    var text = (answerText(p) || res(p).reason || '').trim();
    if (!text) return R(FAIL, 0, 'Empty answer.');
    var qt: string[] = q.match(/[a-z]{4,}/g) || [], at: string[] = text.toLowerCase().match(/[a-z]{4,}/g) || [];
    var onTopic = qt.some(function (t) { return at.indexOf(t) !== -1; });
    var complete = text.split(/\s+/).length >= 6 && '.!?"\')'.indexOf(text.replace(/\s+$/, '').slice(-1)) !== -1;
    if (onTopic && complete) {
      return ALL_CLEAR.test(text)
        ? R(PASS, 1, 'PASS -- No violations reported; answer is on-topic and confident.')
        : R(PASS, 1, 'PASS -- Answer is relevant, coherent and appears correct.');
    }
    return R(FAIL, 0, 'Answer is off-topic or incomplete.');
  }

  function evaluate(payload: Payload): Record<string, EvalResult> {
    var out: Record<string, EvalResult> = {};
    Object.keys(EVALUATORS).forEach(function (k) {
      try { out[k] = EVALUATORS[k](payload); } catch (e) { out[k] = R(INC, null, 'evaluator raised: ' + (e as Error).message); }
    });
    out.deterministic_verdict = deterministicVerdict(out);
    out.llm_judge_surrogate = llmJudgeSurrogate(payload);
    return out;
  }

  // ---------------------------------------------------------------------
  // Parity against the Python run
  // ---------------------------------------------------------------------
  function scoresMatch(a: number | null | undefined, b: number | null | undefined): boolean {
    if (a === null || a === undefined) return b === null || b === undefined;
    return b !== null && b !== undefined && Math.abs(a - b) < 1e-9;
  }

  function selfTest(data: SandboxData): SelfTestResult {
    var total = 0, matched = 0, mismatches: Mismatch[] = [];
    (data.scenarios || []).forEach(function (s) {
      var raw = (s.payload.llm || {} as Partial<LlmResult>).raw_content || '';
      var js = evaluate(runPipeline(s, raw, !s.dropped_by_script));
      Object.keys(s.python_verdicts).forEach(function (k) {
        total += 1;
        var py = s.python_verdicts[k], j = js[k] || {} as Partial<EvalResult>;
        if (py.label === j.label && scoresMatch(py.score, j.score)) matched += 1;
        else mismatches.push({ scenario: s.id, evaluator: k, python: py.label + '/' + py.score, js: j.label + '/' + j.score });
      });
    });
    return { total: total, matched: matched, mismatches: mismatches };
  }

  var api: SandboxApi = {
    runPipeline: runPipeline, evaluate: evaluate, selfTest: selfTest,
    parseResponse: parseResponse, normalizeYear: normalizeYear, HEADLINE: HEADLINE
  };
  root.MuseumQASandbox = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;

  // `node docs/sandbox.js docs/data/sandbox.json` -- parity check from the CLI.
  if (typeof require !== 'undefined' && typeof module !== 'undefined' && require.main === module) {
    var fs = require('fs') as { readFileSync(path: string, encoding: string): string };
    var data = JSON.parse(fs.readFileSync(process.argv[2] || 'docs/data/sandbox.json', 'utf8')) as SandboxData;
    var t = selfTest(data);
    console.log('parity: ' + t.matched + '/' + t.total + ' verdicts match the Python run');
    t.mismatches.forEach(function (m) { console.log('  MISMATCH', JSON.stringify(m)); });
    process.exit(t.mismatches.length ? 1 : 0);
  }
})((typeof window !== 'undefined' ? window : this) as unknown as Root);
