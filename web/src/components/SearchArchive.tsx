import { ArrowDownRight, Loader2, Search, ShieldOff, ShieldCheck } from 'lucide-react';
import { useState, type FormEvent } from 'react';
import { ApiError, retrieve } from '../lib/api';
import type { RetrieveResponse } from '../lib/types';
import type { ApiState } from '../App';
import { Card, Section, Tag } from './ui';

const EXAMPLES: { q: string; why: string }[] = [
  { q: 'When was Omer Fast\'s The Casting made?', why: 'in the archive' },
  { q: 'What medium is Helen Frankenthaler\'s Humming Gold?', why: 'in the archive' },
  { q: 'Reginald Marsh beach etching', why: 'keyword search' },
  { q: 'What is the capital of Mongolia?', why: 'the gate refuses it' },
  { q: 'What is the melting point of gallium arsenide?', why: 'slips past the gate' },
];

export function SearchArchive({ api, onOpenSample }: { api: ApiState; onOpenSample: (id: string) => void }) {
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<RetrieveResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function run(question: string) {
    const text = question.trim();
    if (!text) return;
    setQ(text); setBusy(true); setErr(null);
    try {
      setRes(await retrieve(text));
    } catch (e) {
      setRes(null);
      setErr(e instanceof ApiError ? e.message : 'Search failed.');
    } finally {
      setBusy(false);
    }
  }

  const submit = (e: FormEvent) => { e.preventDefault(); void run(q); };
  const gate = res?.gate;
  const pct = gate ? Math.min(100, (gate.bm25_max / (gate.threshold * 2)) * 100) : 0;

  return (
    <Section id="search" step="Step 1 · live" title="Search 5,010 museum records"
      lead={<>Ask anything. Your question runs through the pipeline's <strong className="text-zinc-200">BM25 retriever</strong> over
        every record from the Cleveland Museum of Art and the Art Institute of Chicago, then through the
        <strong className="text-zinc-200"> abstention gate</strong> that stops off-topic questions before any model is paid for.
        Nothing generates an answer here, and your question is not stored.</>}>
      <Card className="p-4 sm:p-5">
        <form onSubmit={submit} className="flex flex-col sm:flex-row gap-2">
          <label htmlFor="q" className="sr-only">Your question</label>
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" aria-hidden />
            <input id="q" value={q} onChange={(e) => setQ(e.target.value)} maxLength={300}
              placeholder="e.g. Who made Coney Island Beach?"
              className="w-full rounded-xl border border-zinc-700 bg-zinc-950 py-2.5 pl-9 pr-3 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none" />
          </div>
          <button type="submit" disabled={busy || api === 'down' || !q.trim()}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-emerald-500 px-4 py-2.5 text-sm font-semibold text-zinc-950 hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Search className="h-4 w-4" aria-hidden />}
            Search
          </button>
        </form>
        <div className="mt-3 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button key={ex.q} type="button" onClick={() => void run(ex.q)} disabled={busy || api === 'down'}
              className="rounded-full border border-zinc-800 bg-zinc-950 px-3 py-1 text-left text-xs text-zinc-300 hover:border-zinc-600 disabled:opacity-40">
              {ex.q} <span className="text-zinc-500">· {ex.why}</span>
            </button>
          ))}
        </div>
        {api === 'down' && (
          <p className="mt-4 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm text-amber-200">
            Live search needs the API, which is unreachable right now. Step 2 still works: it runs in your browser.
          </p>
        )}
        {api === 'checking' && !res && (
          <p className="mt-4 text-xs text-zinc-500">The API is waking up (it scales to zero when idle); the first search can take a few seconds.</p>
        )}
        {err && <p role="alert" className="mt-4 rounded-lg border border-rose-500/30 bg-rose-500/5 p-3 text-sm text-rose-200">{err}</p>}
      </Card>

      {res && gate && (
        <div className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
          <Card className="p-5">
            <p className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">Abstention gate</p>
            <div className={`mt-3 flex items-center gap-2 text-lg font-semibold ${gate.abstained ? 'text-amber-300' : 'text-emerald-300'}`}>
              {gate.abstained ? <ShieldOff className="h-5 w-5" aria-hidden /> : <ShieldCheck className="h-5 w-5" aria-hidden />}
              {gate.abstained ? 'Stops here, no model call' : 'Passes to the model'}
            </div>
            <div className="mt-4">
              <div className="flex justify-between font-mono text-[11px] text-zinc-500">
                <span>best BM25 {gate.bm25_max}</span><span>floor {gate.threshold}</span>
              </div>
              <div className="relative mt-1.5 h-2 rounded-full bg-zinc-800" role="img"
                aria-label={`Best BM25 score ${gate.bm25_max} against a floor of ${gate.threshold}`}>
                <div className={`h-2 rounded-full ${gate.abstained ? 'bg-amber-400' : 'bg-emerald-400'}`} style={{ width: `${pct}%` }} />
                <div className="absolute top-[-3px] h-[14px] w-0.5 bg-zinc-300" style={{ left: '50%' }} aria-hidden />
              </div>
            </div>
            <p className="mt-4 text-sm leading-relaxed text-zinc-400">{gate.reason}</p>
            {res.recorded_sample_id && (
              <button type="button" onClick={() => onOpenSample(res.recorded_sample_id as string)}
                className="mt-4 inline-flex items-center gap-1.5 text-sm font-medium text-emerald-300 hover:text-emerald-200">
                See Claude's recorded answer to this question <ArrowDownRight className="h-4 w-4" aria-hidden />
              </button>
            )}
            <p className="mt-4 border-t border-zinc-800 pt-3 text-xs leading-relaxed text-zinc-500">{res.note}</p>
            <p className="mt-2 font-mono text-[11px] text-zinc-600">{res.records_searched.toLocaleString()} records · {res.searched_in_ms} ms on the server</p>
          </Card>

          <Card className="divide-y divide-zinc-800">
            <p className="px-5 py-3 font-mono text-[11px] uppercase tracking-wider text-zinc-500">Top {res.candidates.length} records</p>
            <ol>
              {res.candidates.map((c, i) => (
                <li key={c.id} className="px-5 py-3.5 border-t border-zinc-800/70 first:border-t-0">
                  <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                    <span className="font-mono text-xs text-zinc-600">{i + 1}</span>
                    <span className="font-medium text-zinc-100">{c.title ?? 'Untitled'}</span>
                    {c.year && <span className="text-sm text-zinc-500">{c.year}</span>}
                    <span className="ml-auto font-mono text-[11px] text-zinc-500">BM25 {c.bm25_score.toFixed(2)}</span>
                  </div>
                  <p className="mt-0.5 text-sm text-zinc-400">
                    {[c.artist, c.medium].filter(Boolean).join(' · ')}
                  </p>
                  <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                    <Tag tone={c.id.startsWith('cma-') ? 'sky' : 'amber'}>{c.institution ?? c.id}</Tag>
                    <Tag>{c.id}</Tag>
                  </div>
                  {c.snippet && <p className="mt-2 text-xs leading-relaxed text-zinc-500 line-clamp-2">{c.snippet}</p>}
                  {c.attribution && <p className="mt-1 text-[10px] text-zinc-600">{c.attribution}</p>}
                </li>
              ))}
            </ol>
          </Card>
        </div>
      )}
    </Section>
  );
}
