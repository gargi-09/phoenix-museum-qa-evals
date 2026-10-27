// The Cloud Run API (server/app.py). It never calls a model: recorded
// answers are re-scored live, and /retrieve runs live BM25 only.
import type { EvalResult, Health, RetrieveResponse, Scored } from './types';

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? 'https://museum-qa-api-o7gdkkg7ja-uc.a.run.app';

export class ApiError extends Error {
  constructor(message: string, readonly status: number | null) {
    super(message);
  }
}

async function call<T>(path: string, init: RequestInit = {}, timeoutMs = 12000): Promise<T> {
  const ctrl = new AbortController();
  const timer = window.setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(API_BASE + path, {
      ...init,
      signal: ctrl.signal,
      headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
    });
    const body: unknown = await res.json().catch(() => null);
    if (!res.ok) {
      const detail = (body as { detail?: unknown } | null)?.detail;
      throw new ApiError(typeof detail === 'string' ? detail : `HTTP ${res.status}`, res.status);
    }
    return body as T;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError(ctrl.signal.aborted ? 'The API took too long to answer.' : 'The API is unreachable.', null);
  } finally {
    window.clearTimeout(timer);
  }
}

// A cold start (the service scales to zero) measured 6.8 s, so the first
// call gets a longer timeout.
export const health = (): Promise<Health> => call<Health>('/health', {}, 15000);

export const retrieve = (question: string): Promise<RetrieveResponse> =>
  call<RetrieveResponse>('/retrieve', { method: 'POST', body: JSON.stringify({ question }) });

interface EvaluateResponse {
  evaluations: Record<string, EvalResult>;
  deterministic_verdict: EvalResult;
  llm_judge_surrogate: EvalResult;
  evaluated_in_ms: number;
}

export async function scoreOnApi(sampleId: string, answer: string, delivered: boolean): Promise<Scored> {
  const r = await call<EvaluateResponse>('/evaluate', {
    method: 'POST',
    body: JSON.stringify({ sample_id: sampleId, answer, system_prompt_delivered: delivered }),
  }, 8000);
  return { evaluations: r.evaluations, verdict: r.deterministic_verdict, judge: r.llm_judge_surrogate,
           engine: 'api', computeMs: r.evaluated_in_ms };
}
