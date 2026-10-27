// One-click edits that each target one evaluator. They only rewrite the
// recorded answer text; the verdicts come from the real evaluators.
import type { Sample } from './types';

export interface Trick {
  id: string;
  label: string;
  hint: string;
  apply: (raw: string, sample: Sample) => string | null;   // null: not applicable
}

type Parsed = Record<string, unknown> & { answer?: unknown; record_ids_used?: unknown };

function parse(raw: string): Parsed | null {
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start < 0 || end < start) return null;
  try {
    const v: unknown = JSON.parse(raw.slice(start, end + 1));
    return v && typeof v === 'object' && !Array.isArray(v) ? (v as Parsed) : null;
  } catch {
    return null;
  }
}

const dump = (p: Parsed): string => JSON.stringify(p, null, 2);

export const TRICKS: Trick[] = [
  {
    id: 'strip', label: 'Remove the citations', hint: 'Same answer, zero sources. Catches: zero_source_false_pass.',
    apply: (raw) => { const p = parse(raw); return p ? dump({ ...p, record_ids_used: [] }) : null; },
  },
  {
    id: 'fake', label: 'Cite a record never retrieved', hint: 'Adds aic-99999. Catches: citation_groundedness.',
    apply: (raw) => {
      const p = parse(raw);
      if (!p) return null;
      const ids = Array.isArray(p.record_ids_used) ? p.record_ids_used : [];
      return dump({ ...p, record_ids_used: [...ids, 'aic-99999'] });
    },
  },
  {
    id: 'year', label: 'State a wrong year', hint: 'Swaps every year for 1850. Catches: verifier.factual_match.',
    apply: (raw) => {
      const p = parse(raw);
      if (!p || typeof p.answer !== 'string' || !/\b(1[0-9]{3}|20[0-9]{2})\b/.test(p.answer)) return null;
      return dump({ ...p, answer: p.answer.replace(/\b(1[0-9]{3}|20[0-9]{2})\b/g, '1850') });
    },
  },
  {
    id: 'allclear', label: 'Claim “no violations”', hint: 'A confident all-clear with no sources.',
    apply: (raw) => {
      const p = parse(raw);
      return dump({ ...(p ?? {}), answer: 'No violations found. All records are consistent and every field is verified.',
                    answerable: true, record_ids_used: [] });
    },
  },
  {
    id: 'prose', label: 'Answer in plain prose', hint: 'Drops the JSON contract entirely.',
    apply: (raw) => {
      const p = parse(raw);
      const text = p && typeof p.answer === 'string' ? p.answer : 'The work is well documented.';
      return text;
    },
  },
];
