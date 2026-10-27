// The in-browser engine: docs/sandbox.js (compiled from docs/src/sandbox.ts),
// loaded by index.html as a classic script. It is the parity-checked port of
// the Python post-generation pipeline (1,040/1,040 verdicts in
// tests/test_sandbox_parity.py), used whenever the API is unreachable.
import type { EvalResult, Payload, Sample, Scored } from './types';

interface SandboxEngine {
  runPipeline: (scenario: Sample, rawContent: string, delivered: boolean) => Payload;
  evaluate: (payload: Payload) => Record<string, EvalResult>;
  selfTest: (data: { scenarios: Sample[] }) => { total: number; matched: number };
}

declare global {
  interface Window { MuseumQASandbox?: SandboxEngine }
}

export const EVALUATORS = [
  'zero_source_false_pass', 'citation_groundedness', 'confidence_capping', 'prompt_integrity',
  'verifier.citation_membership', 'verifier.not_recorded_violation', 'verifier.factual_match',
  'verifier.not_truncated',
] as const;

function engine(): SandboxEngine {
  const e = window.MuseumQASandbox;
  if (!e) throw new Error('The evaluator engine (sandbox.js) did not load.');
  return e;
}

export function scoreInBrowser(sample: Sample, raw: string, delivered: boolean): Scored {
  const t0 = performance.now();
  const out = engine().evaluate(engine().runPipeline(sample, raw, delivered));
  const evaluations: Record<string, EvalResult> = {};
  for (const k of EVALUATORS) {
    const v = out[k];
    if (v) evaluations[k] = v;
  }
  const verdict = out['deterministic_verdict'];
  const judge = out['llm_judge_surrogate'];
  if (!verdict || !judge) throw new Error('engine returned no verdict');
  return { evaluations, verdict, judge, engine: 'browser', computeMs: round(performance.now() - t0) };
}

export function selfTest(samples: Sample[]): { total: number; matched: number } {
  return engine().selfTest({ scenarios: samples });
}

function round(ms: number): number {
  return Math.round(ms * 100) / 100;
}
