// Shapes of the data the app reads. All of it is produced by scripts in the
// repo (record_samples.py, run_metrics.py, record_llm_judge.py) or returned by
// the API (server/app.py); nothing here is typed in by hand.

export type Label = 'PASS' | 'FAIL' | 'NOT_APPLICABLE' | 'INCONCLUSIVE';

export interface EvalResult {
  label: Label;
  score: number | null;
  explanation: string;
}

export interface ContextBlock {
  id: string;
  fields?: Record<string, string>;
  years?: number[];
  internal_contradiction?: boolean;
}

export interface Payload {
  question: string;
  result?: { abstained?: boolean; reason?: string; [key: string]: unknown };
  candidate_ids?: string[];
  context_blocks?: ContextBlock[];
  prompt_tokens?: number | null;
  llm?: { raw_content?: string } | null;
}

export interface Sample {
  id: string;
  kind: 'recorded' | 'seeded';
  expect: string;
  question: string;
  payload: Payload;
  audit_inputs?: { expected_with_instructions: number; expected_without_instructions: number } | null;
  best_similarity?: number;
  max_tokens?: number;
  dropped_by_script?: boolean;
  seed?: { type: string; from: string; note?: string; transform?: string; record_id?: string } | null;
  recorded?: { model?: string; model_calls?: number; tokens?: number; derived_from?: string };
  python_verdicts: Record<string, EvalResult>;
}

export interface RecordedFile {
  label: string;
  model: string;
  total_model_calls: number;
  total_tokens: number;
  scenarios: Sample[];
}

/** The verdicts for one answer, from either engine. */
export interface Scored {
  evaluations: Record<string, EvalResult>;
  verdict: EvalResult;
  judge: EvalResult;
  engine: 'api' | 'browser';
  computeMs: number | null;
}

export interface RetrieveHit {
  id: string;
  institution: string | null;
  accession_number: string | null;
  title: string | null;
  artist: string | null;
  year: string | null;
  medium: string | null;
  classification: string | null;
  snippet: string;
  attribution: string | null;
  bm25_score: number;
}

export interface RetrieveResponse {
  question: string;
  mode: string;
  records_searched: number;
  gate: { bm25_max: number; threshold: number; abstained: boolean; reason: string };
  candidates: RetrieveHit[];
  note: string;
  searched_in_ms: number;
  recorded_sample_id: string | null;
}

export interface Health {
  status: string;
  samples: number;
  model_calls: string;
  live_retrieval?: string;
  records?: number;
}

interface LatencySummary { n: number; p50_ms: number; p95_ms: number; max_ms: number }

export interface MetricsQuestion {
  id: string;
  kind: string;
  expect: string;
  question: string;
  evaluators: Record<string, Label> | null;
  harness: { outcome: 'accepted' | 'abstained'; abstain_reason: string | null; attempts: number; retried: boolean };
  recorded_tokens: { model_calls: number | null; total_tokens: number | null; prompt_tokens: number | null };
}

export interface Metrics {
  label: string;
  generated_at: string;
  environment: { python: string; platform: string };
  summary: {
    samples: number;
    harness_outcomes: { accepted: number; abstained: number; retried: number };
    abstain_reasons: Record<string, number>;
    latency_ms: {
      first_run_including_imports_ms: number;
      all_runs: LatencySummary;
      per_step: Record<string, LatencySummary>;
      sequential_rescore_evaluate_baseline: LatencySummary;
    };
    recorded_tokens_per_answer: {
      answers_with_a_model_call: number;
      total_tokens: { mean: number; min: number; max: number };
      prompt_tokens: { mean: number; min: number; max: number };
      sum_total_tokens: number;
    } | null;
  };
  questions: MetricsQuestion[];
}

export interface JudgeRecord {
  label: string;
  model: string;
  recorded_at: string;
  summary: { judged: number; judge_passed: string[]; judge_passed_count: number };
  results: {
    id: string;
    expect: string;
    question: string;
    judge: EvalResult;
    deterministic_verdict: Label;
    rule_based_surrogate: Label;
    usage: { total_tokens?: number } | null;
  }[];
}

export interface TraceStep {
  step: 'retrieve' | 'gate' | 'quota' | 'build_prompt' | 'count_tokens' | 'model_call' | 'verify' | 'llm_judge' | 'decide';
  ms: number;
  detail: string;
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
}

interface Usage { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number }

export interface LiveAnswer {
  question: string;
  mode: 'live';
  note: string;
  outcome: 'accepted' | 'abstained' | 'error';
  abstain_reason?: 'retrieval_gate' | 'verification_failed' | null;
  reason?: string;
  error?: string;
  model_called: boolean;
  quota_used?: number;
  bm25_max?: number;
  candidates: { id: string; title: string | null; institution: string | null; bm25_score: number }[];
  trace: TraceStep[];
  total_ms: number;
  usage: { answer: Usage | null; judge: Usage | null; estimated_cost_usd: number; pricing: string };
  output?: { raw: string; answer: string | null; answerable: unknown; record_ids_used: unknown; error: string | null; limitations: unknown };
  confidence?: { score: number | null; label: string } | null;
  prompt_audit?: { status: string; detail: string };
  evaluations?: Record<string, EvalResult>;
  deterministic_verdict?: EvalResult;
  llm_judge_surrogate?: EvalResult;
  llm_judge?: EvalResult;
  context_titles?: Record<string, string | null>;
}

export interface LiveStatus {
  enabled: boolean;
  available?: boolean;
  daily_cap?: number;
  remaining_today?: number | null;
  per_visitor_per_hour?: number;
  visitor_remaining?: number;
  model?: string;
  detail?: string;
}

export type GateBreak = 'none' | 'approve_seeded' | 'drop_known_good' | 'hide_dropped_prompt';

export interface GateRun {
  break: GateBreak;
  break_description: string;
  label: string;
  checks: { ok: boolean; message: string }[];
  release: 'OK' | 'BLOCKED';
  ran_in_ms: number;
  note: string;
}
