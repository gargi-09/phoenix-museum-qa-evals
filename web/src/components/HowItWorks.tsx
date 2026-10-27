import { ArrowRight } from 'lucide-react';
import { useState } from 'react';
import { LINKS } from '../lib/data';
import { Card, ExtLink, Section, Tag } from './ui';

const STEPS: { name: string; detail: string; here: 'live' | 'recorded' | 'partly live' }[] = [
  { name: 'Retrieve', detail: 'BM25 + bge-small embeddings, fused with RRF, entity resolution', here: 'partly live' },
  { name: 'Gate', detail: 'Abstain if BM25 < 19.5 or dense < 0.35: no paid call', here: 'partly live' },
  { name: 'Answer', detail: 'One Claude Haiku call with the retrieved records', here: 'recorded' },
  { name: 'Verify', detail: '8 deterministic evaluators + a prompt-token audit', here: 'live' },
  { name: 'Decide', detail: 'Accept only if all pass; else retry or abstain', here: 'live' },
  { name: 'Trace', detail: 'OpenInference spans into Arize Phoenix', here: 'recorded' },
];

const SHOTS = [
  { src: './assets/trace-zero.png', alt: 'Phoenix trace of the zero-sources case: the judge scores 1.00, the verifiers 0.00' },
  { src: './assets/trace-dropped.png', alt: 'Phoenix trace of the dropped-system-prompt case: prompt_integrity scores 0.00' },
  { src: './assets/phoenix-annotations.png', alt: 'Phoenix span annotations: judge average 1.00 against deterministic verdict 0.25' },
];

export function HowItWorks() {
  return (
    <Section id="how" step="Under the hood" title="How it works"
      lead="The full pipeline runs locally against Arize Phoenix. The public demo runs the parts that cannot spend money or leak a key, and says which is which.">
      <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
        {STEPS.map((s, i) => (
          <li key={s.name} className="relative rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
            <p className="font-mono text-[11px] text-zinc-500">{String(i + 1).padStart(2, '0')}</p>
            <p className="mt-1 font-medium text-zinc-100">{s.name}</p>
            <p className="mt-1 text-xs leading-relaxed text-zinc-400">{s.detail}</p>
            <div className="mt-3"><Tag tone={s.here === 'live' ? 'emerald' : s.here === 'recorded' ? 'zinc' : 'sky'}>here: {s.here}</Tag></div>
            {i < STEPS.length - 1 && <ArrowRight className="absolute -right-2.5 top-1/2 hidden h-4 w-4 -translate-y-1/2 text-zinc-700 lg:block" aria-hidden />}
          </li>
        ))}
      </ol>

      <Card className="mt-5 p-5">
        <p className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">What is live in this demo, and what is recorded</p>
        <ul className="mt-3 grid gap-2 text-sm text-zinc-300 sm:grid-cols-2">
          <li><strong className="text-emerald-300">Live:</strong> BM25 search over all 5,010 records and the BM25 half of the gate (step 1).</li>
          <li><strong className="text-emerald-300">Live:</strong> all 8 evaluators and the harness decision on every edit (step 2).</li>
          <li><strong className="text-zinc-100">Recorded:</strong> 11 real Claude Haiku answers and the records the full hybrid retriever returned for them.</li>
          <li><strong className="text-zinc-100">Not on the public service:</strong> dense retrieval (needs the embedding model) and any model call. The API has no key.</li>
        </ul>
      </Card>

      <p className="mt-8 font-mono text-[11px] uppercase tracking-wider text-zinc-500">From a real Phoenix run</p>
      <div className="mt-3 grid gap-4 md:grid-cols-3">
        {SHOTS.map((s) => <Shot key={s.src} {...s} />)}
      </div>

      <div className="mt-8 flex flex-wrap gap-3 text-sm">
        <ExtLink href={LINKS.design} className="rounded-xl border border-zinc-700 px-4 py-2 text-zinc-200 hover:border-zinc-500">Design doc: failure modes and limits</ExtLink>
        <ExtLink href={LINKS.readme} className="rounded-xl border border-zinc-700 px-4 py-2 text-zinc-200 hover:border-zinc-500">README</ExtLink>
        <a href={LINKS.classic} className="rounded-xl border border-zinc-700 px-4 py-2 text-zinc-200 hover:border-zinc-500">Technical console (trace waterfall)</a>
      </div>
    </Section>
  );
}

function Shot({ src, alt }: { src: string; alt: string }) {
  const [ok, setOk] = useState(true);
  if (!ok) return null;
  return (
    <a href={src} target="_blank" rel="noopener noreferrer" className="block overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900 hover:border-zinc-600">
      <img src={src} alt={alt} loading="lazy" onError={() => setOk(false)} className="h-auto w-full" />
      <p className="p-3 text-xs leading-relaxed text-zinc-500">{alt}</p>
    </a>
  );
}
