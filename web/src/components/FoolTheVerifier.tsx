import { AlertTriangle, CheckCircle2, ChevronDown, Cpu, Loader2, RotateCcw, Scale, ShieldCheck, Wand2 } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { scoreOnApi } from '../lib/api';
import { harnessOutcome, sampleGroup, sampleTitle, type Group } from '../lib/data';
import { EVALUATORS, scoreInBrowser } from '../lib/engine';
import { TRICKS } from '../lib/tricks';
import type { EvalResult, Sample, Scored } from '../lib/types';
import type { ApiState } from '../App';
import { Card, LabelPill, Section, Tag } from './ui';

const WHAT: Record<string, string> = {
  zero_source_false_pass: 'Does it assert something while citing nothing?',
  citation_groundedness: 'Was every cited record actually retrieved?',
  confidence_capping: 'Was confidence capped when the evidence was missing?',
  prompt_integrity: 'Do the billed prompt tokens show the instructions arrived?',
  'verifier.citation_membership': 'Are the citations among the records the model was shown?',
  'verifier.not_recorded_violation': 'Did it fill in a field the record marks “not recorded”?',
  'verifier.factual_match': 'Do the years it states match the cited records?',
  'verifier.not_truncated': 'Was the output cut off?',
};

const GROUPS: { id: Group; title: string }[] = [
  { id: 'seeded', title: 'Seeded failures' },
  { id: 'grounded', title: 'Grounded answers' },
  { id: 'edge', title: 'Edge cases' },
];

export function FoolTheVerifier({ api, samples, selected, onSelect, onApiFailed }: {
  api: ApiState; samples: Sample[]; selected: string; onSelect: (id: string) => void; onApiFailed: () => void;
}) {
  const sample = samples.find((s) => s.id === selected) ?? samples[0];
  const recordedRaw = sample?.payload.llm?.raw_content ?? '';
  const recordedDelivered = !sample?.dropped_by_script;
  const [raw, setRaw] = useState(recordedRaw);
  const [delivered, setDelivered] = useState(recordedDelivered);
  const [scored, setScored] = useState<Scored | null>(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const seq = useRef(0);

  const abstained = Boolean(sample?.payload.result?.abstained);
  const edited = raw !== recordedRaw || delivered !== recordedDelivered;

  useEffect(() => {
    if (!sample || abstained) { setScored(null); return; }
    const mine = ++seq.current;
    const t = window.setTimeout(async () => {
      setBusy(true);
      let out: Scored;
      if (api === 'up') {
        try { out = await scoreOnApi(sample.id, raw, delivered); }
        catch { onApiFailed(); out = scoreInBrowser(sample, raw, delivered); }
      } else {
        out = scoreInBrowser(sample, raw, delivered);
      }
      if (mine === seq.current) { setScored(out); setBusy(false); }
    }, edited ? 350 : 0);
    return () => window.clearTimeout(t);
    // onApiFailed and the derived values are stable for a given sample (the
    // component is keyed by sample id), so only the inputs re-run scoring.
  }, [raw, delivered, api]);

  const byGroup = useMemo(() => GROUPS.map((g) => ({ ...g, items: samples.filter((s) => sampleGroup(s) === g.id) })), [samples]);
  if (!sample) return null;

  const candidates = sample.payload.candidate_ids ?? [];
  const titles = new Map((sample.payload.context_blocks ?? []).map((b) => [b.id, b.fields?.['title'] ?? b.id]));
  const verdict = scored?.verdict ?? null;
  const judge = scored?.judge ?? null;
  const outcome = harnessOutcome(sample, verdict?.label ?? null);
  const disagree = judge?.label === 'PASS' && verdict?.label === 'FAIL';
  const failed = scored ? EVALUATORS.filter((k) => scored.evaluations[k]?.label === 'FAIL') : [];

  return (
    <Section id="verify" step="Step 2 · interactive" title="Try to fool the verifier"
      lead={<>Each sample is a <strong className="text-zinc-200">real Claude Haiku answer</strong>, recorded once with the
        records the full hybrid retriever returned. Edit it however you like: the evaluators re-run on every change.
        A judge that only reads the text sits next to the deterministic verifier, so you can see which one you fooled.</>}>
      <div className="grid gap-5 lg:grid-cols-[280px_minmax(0,1fr)]">
        <div className="lg:hidden">
          <label htmlFor="sample-pick" className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">Pick a sample</label>
          <select id="sample-pick" value={sample.id} onChange={(e) => onSelect(e.target.value)}
            className="mt-1.5 w-full rounded-xl border border-zinc-700 bg-zinc-950 px-3 py-2.5 text-sm text-zinc-100 focus:border-emerald-500 focus:outline-none">
            {byGroup.map((g) => (
              <optgroup key={g.id} label={g.title}>
                {g.items.map((s) => <option key={s.id} value={s.id}>{s.id} · {sampleTitle(s)}</option>)}
              </optgroup>
            ))}
          </select>
        </div>
        <nav aria-label="Samples" className="hidden lg:block space-y-4">
          {byGroup.map((g) => (
            <div key={g.id}>
              <p className="mb-1.5 font-mono text-[11px] uppercase tracking-wider text-zinc-500">{g.title}</p>
              <ul className="space-y-1.5">
                {g.items.map((s) => (
                  <li key={s.id}>
                    <button type="button" onClick={() => onSelect(s.id)} aria-current={s.id === sample.id}
                      className={`w-full rounded-xl border px-3 py-2 text-left transition ${s.id === sample.id
                        ? 'border-emerald-500/50 bg-emerald-500/10' : 'border-zinc-800 bg-zinc-900/40 hover:border-zinc-600'}`}>
                      <span className="flex items-center gap-2">
                        <span className="font-mono text-[11px] text-zinc-500">{s.id}</span>
                        <span className="text-sm font-medium text-zinc-100">{sampleTitle(s)}</span>
                      </span>
                      <span className="mt-0.5 block truncate text-xs text-zinc-500">{s.question}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>

        <div className="min-w-0 space-y-5">
          <Card className="p-5">
            <div className="flex flex-wrap items-center gap-2">
              <Tag tone={sample.kind === 'seeded' ? 'amber' : 'emerald'}>{sample.kind === 'seeded' ? 'seeded failure' : 'recorded answer'}</Tag>
              <Tag>{sample.expect}</Tag>
            </div>
            <p className="mt-3 text-lg font-medium text-zinc-50">{sample.question}</p>
            {sample.seed?.note && <p className="mt-2 text-sm leading-relaxed text-zinc-400">{sample.seed.note}</p>}
            <p className="mt-4 font-mono text-[11px] uppercase tracking-wider text-zinc-500">
              Records the retriever returned ({candidates.length})
            </p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {candidates.length === 0 && <span className="text-sm text-zinc-500">None: the gate stopped this question first.</span>}
              {candidates.map((id) => (
                <span key={id} title={titles.get(id) ?? id}
                  className="max-w-[16rem] truncate rounded-md border border-zinc-800 bg-zinc-950 px-2 py-1 font-mono text-[11px] text-zinc-400">
                  {id}{titles.has(id) ? ` · ${titles.get(id)}` : ''}
                </span>
              ))}
            </div>
          </Card>

          {abstained ? (
            <Card className="p-5">
              <p className="flex items-center gap-2 text-lg font-semibold text-amber-300"><AlertTriangle className="h-5 w-5" aria-hidden />Abstained before any model call</p>
              <p className="mt-2 text-sm leading-relaxed text-zinc-400">{String(sample.payload.result?.reason ?? '')}</p>
              <p className="mt-2 text-sm text-zinc-500">There is no answer to edit: the pipeline spent zero tokens on this question.</p>
            </Card>
          ) : (
            <>
              <Card className="p-5">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">Model output {edited ? '(edited by you)' : '(recorded)'}</p>
                  <button type="button" onClick={() => { setRaw(recordedRaw); setDelivered(recordedDelivered); }} disabled={!edited}
                    className="ml-auto inline-flex items-center gap-1 rounded-lg border border-zinc-700 px-2 py-1 text-xs text-zinc-300 hover:border-zinc-500 disabled:opacity-30">
                    <RotateCcw className="h-3.5 w-3.5" aria-hidden /> Reset
                  </button>
                </div>
                <label htmlFor="raw" className="sr-only">Model output</label>
                <textarea id="raw" value={raw} onChange={(e) => setRaw(e.target.value)} spellCheck={false} maxLength={8000}
                  className="mt-2 h-56 w-full resize-y rounded-xl border border-zinc-800 bg-zinc-950 p-3 font-mono text-xs leading-relaxed text-zinc-200 focus:border-emerald-500 focus:outline-none" />
                <p className="mt-3 flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-wider text-zinc-500"><Wand2 className="h-3.5 w-3.5" aria-hidden />One-click tricks</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  {TRICKS.map((t) => {
                    const next = t.apply(raw, sample);
                    return (
                      <button key={t.id} type="button" title={t.hint} disabled={next === null}
                        onClick={() => next !== null && setRaw(next)}
                        className="rounded-full border border-zinc-700 bg-zinc-950 px-3 py-1 text-xs text-zinc-200 hover:border-emerald-500/60 disabled:opacity-30">
                        {t.label}
                      </button>
                    );
                  })}
                  <label className="inline-flex cursor-pointer items-center gap-2 rounded-full border border-zinc-700 bg-zinc-950 px-3 py-1 text-xs text-zinc-200">
                    <input type="checkbox" checked={!delivered} onChange={(e) => setDelivered(!e.target.checked)} className="accent-emerald-500" />
                    Drop the system prompt
                  </label>
                </div>
              </Card>

              {disagree && (
                <div role="status" className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-100">
                  <strong>The judge was fooled.</strong> It passed an answer the verifier failed: exactly the false pass this project exists to catch.
                </div>
              )}

              <div className="grid gap-4 sm:grid-cols-2">
                <VerdictCard icon={<Scale className="h-4 w-4" aria-hidden />} title="Judge (reads the text only)"
                  sub="A rule-based stand-in for an LLM judge: it sees the question and the answer, nothing else."
                  result={judge} busy={busy} />
                <VerdictCard icon={<ShieldCheck className="h-4 w-4" aria-hidden />} title="Deterministic verifier"
                  sub="Sees the citations, the retrieved records and the billed prompt tokens."
                  result={verdict} busy={busy} />
              </div>

              <Card className="p-5">
                <div className="flex flex-wrap items-center gap-3">
                  <p className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">Harness decision</p>
                  <span className={`inline-flex items-center gap-1.5 text-sm font-semibold ${outcome.tone === 'ok' ? 'text-emerald-300' : 'text-amber-300'}`}>
                    {outcome.tone === 'ok' ? <CheckCircle2 className="h-4 w-4" aria-hidden /> : <AlertTriangle className="h-4 w-4" aria-hidden />}
                    {outcome.label}
                  </span>
                  {scored && (
                    <span className="ml-auto inline-flex items-center gap-1.5 font-mono text-[11px] text-zinc-500">
                      <Cpu className="h-3.5 w-3.5" aria-hidden />
                      {scored.engine === 'api' ? `Python evaluators on the live API · ${scored.computeMs} ms` : `TypeScript port in your browser · ${scored.computeMs} ms`}
                    </span>
                  )}
                </div>
                <p className="mt-2 text-xs leading-relaxed text-zinc-500">
                  The answer is accepted only if all 8 checks pass. Each sample has one recorded answer, so a failure ends in
                  “abstain” here; the retry path is covered by the project's tests.
                </p>
                <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}
                  className="mt-4 inline-flex items-center gap-1.5 text-sm font-medium text-zinc-200 hover:text-zinc-50">
                  <ChevronDown className={`h-4 w-4 transition ${open ? 'rotate-180' : ''}`} aria-hidden />
                  {open ? 'Hide' : 'Show'} all 8 checks{failed.length ? ` · ${failed.length} failing` : ''}
                </button>
                {open && scored && (
                  <ul className="mt-3 divide-y divide-zinc-800 rounded-xl border border-zinc-800">
                    {EVALUATORS.map((k) => {
                      const r = scored.evaluations[k];
                      return r ? (
                        <li key={k} className="p-3">
                          <div className="flex flex-wrap items-center gap-2">
                            <LabelPill label={r.label} />
                            <span className="font-mono text-xs text-zinc-300">{k}</span>
                            <span className="text-xs text-zinc-500">{WHAT[k]}</span>
                          </div>
                          <p className="mt-1.5 break-words text-xs leading-relaxed text-zinc-400">{r.explanation}</p>
                        </li>
                      ) : null;
                    })}
                  </ul>
                )}
              </Card>
            </>
          )}
        </div>
      </div>
    </Section>
  );
}

function VerdictCard({ icon, title, sub, result, busy }: {
  icon: ReactNode; title: string; sub: string; result: EvalResult | null; busy: boolean;
}) {
  return (
    <Card className="p-5">
      <p className="flex items-center gap-2 text-sm font-medium text-zinc-200">{icon}{title}</p>
      <p className="mt-1 text-xs leading-relaxed text-zinc-500">{sub}</p>
      <div className="mt-4 flex items-center gap-3">
        {result ? <LabelPill label={result.label} big /> : <span className="h-7 w-16 rounded-full bg-zinc-800" />}
        {busy && <Loader2 className="h-4 w-4 animate-spin text-zinc-500" aria-label="Scoring" />}
      </div>
      {result && <p className="mt-3 line-clamp-4 break-words text-xs leading-relaxed text-zinc-400">{result.explanation}</p>}
    </Card>
  );
}
