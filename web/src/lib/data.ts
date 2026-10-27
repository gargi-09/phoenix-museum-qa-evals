import type { JudgeRecord, Metrics, RecordedFile, Sample } from './types';

async function json<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

export const loadRecorded = (): Promise<RecordedFile> => json<RecordedFile>('./data/recorded_samples.json');
export const loadMetrics = (): Promise<Metrics> => json<Metrics>('./data/metrics.json');
export const loadJudge = (): Promise<JudgeRecord> => json<JudgeRecord>('./data/recorded_llm_judge.json');

export const REPO = 'https://github.com/gargi-09/phoenix-museum-qa-evals';
export const LINKS = {
  repo: REPO,
  design: `${REPO}/blob/main/docs/DESIGN.md`,
  readme: `${REPO}#readme`,
  linkedin: 'https://www.linkedin.com/in/gargi-vipat/',
  portfolio: 'https://gargi-vipat.framer.website/',
  classic: './classic.html',
};

/** Friendly names for the sample picker; the source of truth stays `expect`. */
export function sampleTitle(s: Sample): string {
  const titles: Record<string, string> = {
    s01: 'Zero sources, confident answer',
    s02: 'System prompt silently dropped',
    s03: 'Cites a record it never retrieved',
    q06: 'Correct answer, verifier false positive',
    q08: 'Correct decline: records are silent',
    q09: 'Correct decline: an opinion question',
    q10: 'Off-topic, slips past the gate',
    q11: 'Off-topic, stopped by the gate',
  };
  return titles[s.id] ?? 'Grounded answer';
}

export type Group = 'seeded' | 'grounded' | 'edge';

export function sampleGroup(s: Sample): Group {
  if (s.kind === 'seeded') return 'seeded';
  return ['q06', 'q08', 'q09', 'q10', 'q11'].includes(s.id) ? 'edge' : 'grounded';
}

export function harnessOutcome(s: Sample, verdict: string | null): { label: string; tone: 'ok' | 'stop' } {
  if (s.payload.result?.abstained) return { label: 'Abstained at the retrieval gate · no model call', tone: 'stop' };
  if (verdict === 'PASS') return { label: 'Accepted', tone: 'ok' };
  return { label: 'Abstained · no further recorded attempt', tone: 'stop' };
}
