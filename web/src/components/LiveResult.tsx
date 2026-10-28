import { AlertTriangle, CheckCircle2, ChevronDown, Scale, ShieldCheck, ShieldOff } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { EVALUATORS } from '../lib/engine';
import type { EvalResult, LiveAnswer, TraceStep } from '../lib/types';
import { Card, LabelPill, Tag } from './ui';

const STEP_LABEL: Record<TraceStep['step'], string> = {
  retrieve: 'Retrieve (BM25, live)',
  gate: 'Abstention gate',
  quota: 'Quota check',
  build_prompt: 'Build context + prompt',
  count_tokens: 'Count tokens (audit inputs)',
  model_call: 'Claude Haiku: answer',
  verify: '8 deterministic checks',
  llm_judge: 'Claude Haiku: LLM judge',
  decide: 'Harness decision',
};

const PAID: TraceStep['step'][] = ['model_call', 'llm_judge'];

export function LiveResult({ r }: { r: LiveAnswer }) {
  const [open, setOpen] = useState(false);
  const verdict = r.deterministic_verdict;
  const judge = r.llm_judge;
  const fooled = judge?.label === 'PASS' && verdict?.label === 'FAIL';
  const accepted = r.outcome === 'accepted';
  const cited = Array.isArray(r.output?.record_ids_used) ? (r.output?.record_ids_used as unknown[]).map(String) : [];

  return (
    <div className="mt-5 space-y-4">
      <Card className="p-5">
        <div className="flex flex-wrap items-center gap-2">
          <Tag tone="emerald">live · {r.model_called ? 'real model call' : 'no model call'}</Tag>
          <span className={`inline-flex items-center gap-1.5 text-sm font-semibold ${accepted ? 'text-emerald-300' : 'text-amber-300'}`}>
            {accepted ? <CheckCircle2 className="h-4 w-4" aria-hidden /> : r.abstain_reason === 'retrieval_gate'
              ? <ShieldOff className="h-4 w-4" aria-hidden /> : <AlertTriangle className="h-4 w-4" aria-hidden />}
            {accepted ? 'Accepted: every check passed'
              : r.abstain_reason === 'retrieval_gate' ? 'Abstained at the gate: no model call, no cost'
              : r.outcome === 'error' ? `The model call failed (${r.error ?? 'error'})`
              : 'Abstained: the verifier failed this answer'}
          </span>
          <span className="ml-auto font-mono text-[11px] text-zinc-500">{(r.total_ms / 1000).toFixed(2)} s end to end</span>
        </div>

        {r.output?.answer && (
          <>
            <p className="mt-4 font-mono text-[11px] uppercase tracking-wider text-zinc-500">Claude's answer</p>
            <p className="mt-1.5 text-base leading-relaxed text-zinc-100">{r.output.answer}</p>
          </>
        )}
        {!r.output?.answer && r.output?.raw && (
          <>
            <p className="mt-4 font-mono text-[11px] uppercase tracking-wider text-zinc-500">Raw model output (did not parse)</p>
            <pre className="mt-1.5 max-h-40 overflow-auto whitespace-pre-wrap rounded-lg border border-zinc-800 bg-zinc-950 p-3 font-mono text-xs text-zinc-300">{r.output.raw}</pre>
          </>
        )}
        {r.reason && <p className="mt-3 text-sm text-zinc-400">{r.reason}</p>}
        {cited.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            <span className="text-xs text-zinc-500">Cites:</span>
            {cited.map((id) => (
              <span key={id} className="rounded-md border border-zinc-800 bg-zinc-950 px-2 py-0.5 font-mono text-[11px] text-zinc-400">
                {id}{r.context_titles?.[id] ? ` · ${r.context_titles[id]}` : ' · not among the retrieved records'}
              </span>
            ))}
          </div>
        )}
      </Card>

      {fooled && (
        <div role="status" className="rounded-xl border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-100">
          <strong>The LLM judge was fooled, live.</strong> It passed an answer the deterministic verifier failed.
        </div>
      )}

      {r.model_called && verdict && judge && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Verdict icon={<Scale className="h-4 w-4" aria-hidden />} title="Real LLM judge (live)"
            sub="claude-haiku-5-5, given the question and answer only." result={judge} />
          <Verdict icon={<ShieldCheck className="h-4 w-4" aria-hidden />} title="Deterministic verifier"
            sub="Citations, retrieved records, billed prompt tokens." result={verdict} />
        </div>
      )}

      <Card className="p-5">
        <div className="flex flex-wrap items-baseline gap-2">
          <p className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">Trace</p>
          <span className="text-xs text-zinc-500">every step this request ran, with its real duration</span>
        </div>
        <Waterfall steps={r.trace} total={r.total_ms} />
        <dl className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4 text-xs">
          <Fact k="Prompt tokens billed" v={r.usage.answer?.prompt_tokens?.toLocaleString() ?? '–'} />
          <Fact k="Output tokens" v={r.usage.answer?.completion_tokens?.toLocaleString() ?? '–'} />
          <Fact k="Prompt audit" v={r.prompt_audit?.status ?? '–'} />
          <Fact k="Cost (estimate)" v={r.model_called ? `$${r.usage.estimated_cost_usd.toFixed(5)}` : '$0'} />
        </dl>
        <p className="mt-3 text-[11px] leading-relaxed text-zinc-600">{r.usage.pricing}, answer + judge. {r.note}</p>

        {r.evaluations && (
          <>
            <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}
              className="mt-4 inline-flex items-center gap-1.5 text-sm font-medium text-zinc-200 hover:text-zinc-50">
              <ChevronDown className={`h-4 w-4 transition ${open ? 'rotate-180' : ''}`} aria-hidden />
              {open ? 'Hide' : 'Show'} all 8 checks
            </button>
            {open && (
              <ul className="mt-3 divide-y divide-zinc-800 rounded-xl border border-zinc-800">
                {EVALUATORS.map((k) => {
                  const e = r.evaluations?.[k];
                  return e ? (
                    <li key={k} className="p-3">
                      <div className="flex items-center gap-2"><LabelPill label={e.label} /><span className="font-mono text-xs text-zinc-300">{k}</span></div>
                      <p className="mt-1.5 break-words text-xs leading-relaxed text-zinc-400">{e.explanation}</p>
                    </li>
                  ) : null;
                })}
              </ul>
            )}
          </>
        )}
      </Card>
    </div>
  );
}

function Waterfall({ steps, total }: { steps: TraceStep[]; total: number }) {
  let start = 0;
  return (
    <ol className="mt-3 space-y-1.5">
      {steps.map((s) => {
        const left = total > 0 ? (start / total) * 100 : 0;
        const width = total > 0 ? Math.max(0.6, (s.ms / total) * 100) : 0;
        start += s.ms;
        const paid = PAID.includes(s.step);
        return (
          <li key={s.step} className="grid grid-cols-[minmax(0,10.5rem)_minmax(0,1fr)_4.5rem] items-center gap-2 text-xs sm:grid-cols-[13rem_minmax(0,1fr)_5rem]">
            <span className="truncate text-zinc-300" title={s.detail}>{STEP_LABEL[s.step]}</span>
            <span className="relative h-2.5 rounded bg-zinc-800/70" title={s.detail}>
              <span className={`absolute top-0 h-2.5 rounded ${paid ? 'bg-sky-400' : 'bg-emerald-400/80'}`}
                style={{ left: `${Math.min(left, 99.4)}%`, width: `${Math.min(width, 100 - Math.min(left, 99.4))}%` }} />
            </span>
            <span className="text-right font-mono text-zinc-500">{s.ms < 10 ? s.ms.toFixed(1) : Math.round(s.ms)} ms</span>
          </li>
        );
      })}
    </ol>
  );
}

function Verdict({ icon, title, sub, result }: { icon: ReactNode; title: string; sub: string; result: EvalResult }) {
  return (
    <Card className="p-5">
      <p className="flex items-center gap-2 text-sm font-medium text-zinc-200">{icon}{title}</p>
      <p className="mt-1 text-xs text-zinc-500">{sub}</p>
      <div className="mt-3"><LabelPill label={result.label} big /></div>
      <p className="mt-3 line-clamp-4 break-words text-xs leading-relaxed text-zinc-400">{result.explanation}</p>
    </Card>
  );
}

function Fact({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <dt className="text-zinc-500">{k}</dt>
      <dd className="mt-0.5 font-mono text-sm text-zinc-100">{v}</dd>
    </div>
  );
}
