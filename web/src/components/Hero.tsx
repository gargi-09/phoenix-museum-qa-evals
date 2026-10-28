import { ArrowRight, Cpu, GitCompare, ShieldAlert, Timer } from 'lucide-react';
import type { ComponentType } from 'react';
import { LINKS } from '../lib/data';
import { ExtLink } from './ui';

// Every value here is measured or recorded in the repo; the note says where.
const TILES: { icon: ComponentType<{ className?: string }>; value: string; label: string; note: string }[] = [
  { icon: ShieldAlert, value: '3/3', label: 'Seeded false passes blocked',
    note: 'The real LLM judge approved 2 of them; the rule-based stand-in approved all 3.' },
  { icon: GitCompare, value: '1,040/1,040', label: 'Browser–Python parity',
    note: '104 edited answers scored by the TypeScript port and the Python pipeline.' },
  { icon: Timer, value: '34 ms', label: 'Live search, server time',
    note: 'BM25 over 5,010 records on Cloud Run. Round trip from one laptop: p50 362 ms.' },
  { icon: Cpu, value: '0', label: 'Extra tokens to verify',
    note: '8 deterministic evaluators. No second model call.' },
];

export function Hero() {
  return (
    <section id="top" className="mx-auto max-w-6xl px-4 sm:px-6 pt-14 sm:pt-20 pb-6">
      <p className="font-mono text-xs uppercase tracking-widest text-emerald-400/90">Arize Phoenix · OpenInference · Claude Haiku</p>
      <h1 className="mt-3 max-w-4xl text-3xl sm:text-5xl font-semibold tracking-tight leading-[1.1] text-zinc-50">
        An LLM judge reads the answer.<br className="hidden sm:block" />{' '}
        <span className="text-emerald-400">A verifier reads the evidence.</span>
      </h1>
      <p className="mt-5 max-w-2xl text-base sm:text-lg leading-relaxed text-zinc-400">
        A question-answering pipeline over 5,010 public museum records. A fluent, confident answer
        that cites <em>nothing</em> fools a judge that only reads the text. Eight deterministic checks
        look at what it cited, what was retrieved and what the API billed, and catch it.
      </p>
      <div className="mt-7 flex flex-wrap gap-3">
        <a href="#search" className="inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-4 py-2.5 text-sm font-semibold text-zinc-950 hover:bg-emerald-400">
          Try it <ArrowRight className="h-4 w-4" aria-hidden />
        </a>
        <ExtLink href={LINKS.repo} className="inline-flex items-center gap-2 rounded-xl border border-zinc-700 px-4 py-2.5 text-sm font-medium text-zinc-200 hover:border-zinc-500">
          Read the code
        </ExtLink>
        <ExtLink href={LINKS.design} className="inline-flex items-center gap-2 rounded-xl border border-zinc-700 px-4 py-2.5 text-sm font-medium text-zinc-200 hover:border-zinc-500">
          Design doc
        </ExtLink>
      </div>

      <ol className="mt-10 grid gap-3 sm:grid-cols-3 text-sm">
        {[
          ['1', 'Ask your own question', 'A real Claude answer, verified live by 8 checks and judged by an LLM, with the full trace.'],
          ['2', 'Try to fool the verifier', 'Edit a real Claude answer. Strip its sources, fake a citation, drop the system prompt.'],
          ['3', 'Run the release gate', 'The CI gate, live. Break something on purpose and watch it refuse to ship.'],
        ].map(([n, t, d]) => (
          <li key={n} className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
            <p className="font-mono text-xs text-emerald-400">Step {n}</p>
            <p className="mt-1 font-medium text-zinc-100">{t}</p>
            <p className="mt-1 text-zinc-400 leading-relaxed">{d}</p>
          </li>
        ))}
      </ol>

      <div className="mt-6 grid gap-3 grid-cols-1 sm:grid-cols-2 lg:grid-cols-4">
        {TILES.map(({ icon: Icon, value, label, note }) => (
          <div key={label} className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
            <p className="flex items-center gap-2 font-mono text-[11px] uppercase tracking-wider text-zinc-500">
              <Icon className="h-3.5 w-3.5" />{label}
            </p>
            <p className="mt-2 font-mono text-2xl sm:text-3xl font-semibold text-zinc-50">{value}</p>
            <p className="mt-1 text-xs leading-relaxed text-zinc-500">{note}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
