import { ArrowDownRight, Check, History, Loader2, Scale, ShieldCheck, User } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { scoreOnApi } from '../lib/api';
import { harnessOutcome } from '../lib/data';
import { scoreInBrowser } from '../lib/engine';
import type { Sample, Scored } from '../lib/types';
import type { ApiState } from '../App';
import { LabelPill } from './ui';

// A recorded Claude answer, replayed like a chat turn. The steps and the
// typing are presentation; the label says it is a replay. The 8 checks are
// NOT replayed: they run now, on the API (or in the browser if it is down).

type StepState = 'wait' | 'run' | 'done';

const reduceMotion = (): boolean =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export function RecordedReplay({ sample, api, onOpenSample }: {
  sample: Sample; api: ApiState; onOpenSample: (id: string) => void;
}) {
  const raw = sample.payload.llm?.raw_content ?? '';
  const answer = extractAnswer(raw);
  const abstained = Boolean(sample.payload.result?.abstained);
  const candidates = sample.payload.candidate_ids?.length ?? 0;
  const when = sample.recorded?.recorded_at?.slice(0, 10);

  const steps = abstained
    ? [`Searched 5,010 records`, 'Abstention gate: stopped before any model call']
    : [`Searched 5,010 records · ${candidates} candidates`, 'Abstention gate: passed',
       `Claude Haiku answered${when ? ` (recorded ${when})` : ''}`, 'Running the 8 checks, live now'];

  const [stage, setStage] = useState(0);            // how many steps are done
  const [typed, setTyped] = useState(0);
  const [scored, setScored] = useState<Scored | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    const instant = reduceMotion();
    if (!abstained) {
      const delivered = !sample.dropped_by_script;
      (api === 'up' ? scoreOnApi(sample.id, raw, delivered).catch(() => scoreInBrowser(sample, raw, delivered))
                    : Promise.resolve(scoreInBrowser(sample, raw, delivered)))
        .then((s) => { if (alive.current) setScored(s); });
    }
    if (instant) { setStage(steps.length); setTyped(answer.length); return () => { alive.current = false; }; }
    const timers: number[] = [];
    steps.forEach((_, i) => timers.push(window.setTimeout(() => alive.current && setStage(i + 1), 450 + i * 550)));
    const startTyping = 450 + steps.length * 550;
    timers.push(window.setTimeout(() => {
      const t = window.setInterval(() => {
        if (!alive.current) return window.clearInterval(t);
        setTyped((n) => { if (n >= answer.length) { window.clearInterval(t); return n; } return Math.min(answer.length, n + 3); });
      }, 16);
      timers.push(t);
    }, startTyping));
    return () => { alive.current = false; timers.forEach((t) => { window.clearTimeout(t); window.clearInterval(t); }); };
    // Keyed by sample id by the parent: one replay per mount.
  }, [sample.id]);

  const stepState = (i: number): StepState => (i < stage ? 'done' : i === stage ? 'run' : 'wait');
  const doneTyping = typed >= answer.length && stage >= steps.length;
  const outcome = harnessOutcome(sample, scored?.verdict.label ?? null);

  return (
    <div className="mt-5 space-y-3" aria-live="polite">
      <div className="flex justify-end">
        <div className="flex max-w-[85%] items-start gap-2 rounded-2xl rounded-tr-sm bg-emerald-500/15 px-4 py-2.5 text-sm text-emerald-50">
          <span>{sample.question}</span><User className="mt-0.5 h-4 w-4 shrink-0 text-emerald-300" aria-hidden />
        </div>
      </div>

      <div className="max-w-[92%] rounded-2xl rounded-tl-sm border border-zinc-800 bg-zinc-900/70 p-4">
        <span className="inline-flex items-center gap-1.5 rounded-full border border-zinc-700 px-2 py-0.5 font-mono text-[11px] text-zinc-400">
          <History className="h-3 w-3" aria-hidden /> recorded answer · replayed
        </span>
        <ol className="mt-3 space-y-1.5 text-sm">
          {steps.map((label, i) => {
            const st = stepState(i);
            return (
              <li key={label} className={`flex items-center gap-2 ${st === 'wait' ? 'text-zinc-600' : 'text-zinc-300'}`}>
                {st === 'done' ? <Check className="h-4 w-4 text-emerald-400" aria-hidden />
                  : st === 'run' ? <Loader2 className="h-4 w-4 animate-spin text-zinc-400" aria-hidden />
                  : <span className="h-4 w-4 rounded-full border border-zinc-700" aria-hidden />}
                {label}
              </li>
            );
          })}
        </ol>

        {abstained ? (
          stage >= steps.length && (
            <p className="mt-3 text-sm leading-relaxed text-amber-200">{String(sample.payload.result?.reason ?? 'Abstained.')} No tokens were spent.</p>
          )
        ) : (
          stage >= steps.length - 1 && (
            <p className="mt-3 whitespace-pre-wrap text-[15px] leading-relaxed text-zinc-100">
              {answer.slice(0, typed)}{!doneTyping && <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-zinc-400 align-middle" aria-hidden />}
            </p>
          )
        )}

        {doneTyping && !abstained && (
          <div className="mt-4 border-t border-zinc-800 pt-3">
            {scored ? (
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
                <span className="inline-flex items-center gap-1.5 text-zinc-400"><Scale className="h-3.5 w-3.5" aria-hidden />Judge (rule-based) <LabelPill label={scored.judge.label} /></span>
                <span className="inline-flex items-center gap-1.5 text-zinc-400"><ShieldCheck className="h-3.5 w-3.5" aria-hidden />Verifier <LabelPill label={scored.verdict.label} /></span>
                <span className={outcome.tone === 'ok' ? 'text-emerald-300' : 'text-amber-300'}>{outcome.label}</span>
                <span className="font-mono text-zinc-600">checks: {scored.engine === 'api' ? 'Python on the live API' : 'TypeScript in your browser'}</span>
              </div>
            ) : <p className="text-xs text-zinc-500">Running the checks…</p>}
            <button type="button" onClick={() => onOpenSample(sample.id)}
              className="mt-3 inline-flex items-center gap-1.5 text-sm font-medium text-emerald-300 hover:text-emerald-200">
              Edit this answer and try to fool the verifier <ArrowDownRight className="h-4 w-4" aria-hidden />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function extractAnswer(raw: string): string {
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start >= 0 && end > start) {
    try {
      const v: unknown = JSON.parse(raw.slice(start, end + 1));
      const a = v && typeof v === 'object' ? (v as { answer?: unknown }).answer : undefined;
      if (typeof a === 'string' && a.trim()) return a;
    } catch { /* not JSON: show the raw text, as the pipeline would */ }
  }
  return raw.trim();
}
