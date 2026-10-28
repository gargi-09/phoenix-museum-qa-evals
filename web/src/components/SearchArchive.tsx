import { ArrowDownRight, Loader2, Search, ShieldOff, ShieldCheck, Sparkles } from 'lucide-react';
import { useEffect, useState, type FormEvent } from 'react';
import { ApiError, liveAnswer, liveStatus, retrieve } from '../lib/api';
import type { LiveAnswer, LiveStatus, RetrieveResponse, Sample } from '../lib/types';
import type { ApiState } from '../App';
import { LiveResult } from './LiveResult';
import { RecordedReplay } from './RecordedReplay';
import { Card, Section, Tag } from './ui';

const EXAMPLES: { q: string; why: string }[] = [
  { q: 'When was Omer Fast\'s The Casting made?', why: 'in the archive' },
  { q: 'What medium is Helen Frankenthaler\'s Humming Gold?', why: 'in the archive' },
  { q: 'Reginald Marsh beach etching', why: 'keyword search' },
  { q: 'What is the capital of Mongolia?', why: 'the gate refuses it' },
  { q: 'What is the melting point of gallium arsenide?', why: 'slips past the gate' },
];

// Recorded questions offered for instant replay (real Claude answers).
const REPLAY_IDS = ['q01', 'q03', 'q05', 'q07', 'q09', 'q10', 'q11'];

export function SearchArchive({ api, samples, onOpenSample }: {
  api: ApiState; samples: Sample[]; onOpenSample: (id: string) => void;
}) {
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<RetrieveResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [answer, setAnswer] = useState<LiveAnswer | null>(null);
  const [answering, setAnswering] = useState(false);
  const [quota, setQuota] = useState<LiveStatus | null>(null);
  const [replay, setReplay] = useState<Sample | null>(null);
  const [unavailable, setUnavailable] = useState<string | null>(null);
  const replayable = samples.filter((s) => REPLAY_IDS.includes(s.id));
  const recordedFor = (text: string): Sample | undefined =>
    samples.find((s) => s.kind === 'recorded' && s.question.trim().toLowerCase() === text.trim().toLowerCase());
  const startReplay = (s: Sample) => { setQ(s.question); setReplay(s); setAnswer(null); setRes(null); setErr(null); };

  const refreshQuota = () => { liveStatus().then(setQuota).catch(() => setQuota(null)); };
  useEffect(() => { if (api === 'up') refreshQuota(); }, [api]);

  async function run(question: string) {
    const text = question.trim();
    if (!text) return;
    setQ(text); setBusy(true); setErr(null); setAnswer(null); setReplay(null);
    try {
      setRes(await retrieve(text));
    } catch (e) {
      setRes(null);
      setErr(e instanceof ApiError ? e.message : 'Search failed.');
    } finally {
      setBusy(false);
    }
  }

  async function ask(question: string) {
    const text = question.trim();
    if (!text) return;
    setQ(text); setAnswering(true); setErr(null); setRes(null); setReplay(null); setUnavailable(null);
    try {
      setAnswer(await liveAnswer(text));
    } catch (e) {
      setAnswer(null);
      const msg = e instanceof ApiError ? e.message : 'The live answer failed.';
      // Quota used up, credit run out, key not working, model busy: point to the recordings.
      if (e instanceof ApiError && (e.status === 429 || e.status === 503)) setUnavailable(msg);
      else setErr(msg);
    } finally {
      setAnswering(false);
      refreshQuota();
    }
  }

  const liveOn = api === 'up' && quota?.enabled === true;
  const left = quota?.remaining_today ?? null;
  const visitorLeft = quota?.visitor_remaining ?? null;
  const paused = quota?.paused_reason ? (quota.message ?? 'Live answers are paused right now.') : null;
  const canAsk = liveOn && !answering && !busy && !!q.trim() && left !== 0 && visitorLeft !== 0 && quota?.available !== false;
  const notice = unavailable ?? paused ?? (liveOn && left === 0 ? "Today's live answers are used up." : null);
  const submit = (e: FormEvent) => { e.preventDefault(); void run(q); };
  const gate = res?.gate;
  const pct = gate ? Math.min(100, (gate.bm25_max / (gate.threshold * 2)) * 100) : 0;

  return (
    <Section id="search" step="Step 1 · live" title="Ask the archive"
      lead={<>Ask anything about 5,010 records from the Cleveland Museum of Art and the Art Institute of Chicago.
        <strong className="text-zinc-200"> Search</strong> shows what the retriever finds and whether the
        <strong className="text-zinc-200"> abstention gate</strong> lets the question through (free, no model).
        <strong className="text-zinc-200"> Answer it</strong> runs the whole pipeline live: one real Claude Haiku call, all 8
        checks, and a real LLM judge, with the trace of every step. Your question is not stored or logged.</>}>
      <Card className="p-4 sm:p-5">
        <form onSubmit={submit} className="flex flex-col sm:flex-row gap-2">
          <label htmlFor="q" className="sr-only">Your question</label>
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" aria-hidden />
            <input id="q" value={q} onChange={(e) => setQ(e.target.value)} maxLength={300}
              placeholder="e.g. Who made Coney Island Beach?"
              className="w-full rounded-xl border border-zinc-700 bg-zinc-950 py-2.5 pl-9 pr-3 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-emerald-500 focus:outline-none" />
          </div>
          <button type="submit" disabled={busy || answering || api === 'down' || !q.trim()}
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-zinc-600 px-4 py-2.5 text-sm font-semibold text-zinc-100 hover:border-zinc-400 disabled:cursor-not-allowed disabled:opacity-40">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Search className="h-4 w-4" aria-hidden />}
            Search
          </button>
          <button type="button" onClick={() => void ask(q)} disabled={!canAsk}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-emerald-500 px-4 py-2.5 text-sm font-semibold text-zinc-950 hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-40">
            {answering ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Sparkles className="h-4 w-4" aria-hidden />}
            {answering ? 'Answering…' : 'Answer it (live)'}
          </button>
        </form>
        <p className="mt-2 font-mono text-[11px] text-zinc-500">
          {liveOn
            ? left === 0 ? "Today's live answers are used up. Search, step 2 and the release gate still work."
              : visitorLeft === 0 ? 'You have used your live answers for this hour. Search still works.'
              : `Live answers left today: ${left ?? '?'} of ${quota?.daily_cap ?? '?'} · you: ${visitorLeft ?? '?'} this hour · ${quota?.model ?? ''}`
            : api === 'up' && quota && !quota.enabled ? 'Live answers are off right now. Search, step 2 and the release gate still work.'
            : ''}
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button key={ex.q} type="button" onClick={() => setQ(ex.q)} disabled={busy || answering}
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
        {notice && (
          <div role="status" className="mt-4 rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
            <p className="text-sm text-amber-100">{notice}</p>
            <p className="mt-1 text-sm text-zinc-400">These are real Claude answers recorded earlier. Pick one to replay it; the 8 checks run on it live.</p>
          </div>
        )}
        <div className="mt-4">
          <p className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">Or replay a recorded Claude answer</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {replayable.map((s) => (
              <button key={s.id} type="button" onClick={() => startReplay(s)}
                className={`rounded-full border px-3 py-1 text-left text-xs hover:border-emerald-500/60 ${replay?.id === s.id
                  ? 'border-emerald-500/60 bg-emerald-500/10 text-emerald-100' : notice ? 'border-amber-500/40 bg-zinc-950 text-zinc-200' : 'border-zinc-800 bg-zinc-950 text-zinc-300'}`}>
                {s.question}
              </button>
            ))}
          </div>
        </div>
        {answering && (
          <p className="mt-4 text-sm text-zinc-400">Retrieving, checking the gate, calling Claude, verifying, asking the judge… (a few seconds)</p>
        )}
      </Card>

      {answer && <LiveResult r={answer} />}
      {replay && <RecordedReplay key={replay.id} sample={replay} api={api} onOpenSample={onOpenSample} />}
      {!replay && !answer && unavailable && recordedFor(q) && (
        <button type="button" onClick={() => { const s = recordedFor(q); if (s) startReplay(s); }}
          className="mt-4 text-sm font-medium text-emerald-300 hover:text-emerald-200">
          This exact question has a recorded answer: replay it
        </button>
      )}

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
