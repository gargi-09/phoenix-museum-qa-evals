import { CheckCircle2, Loader2, Play, XCircle } from 'lucide-react';
import { useState } from 'react';
import { ApiError, runGate } from '../lib/api';
import type { GateBreak, GateRun } from '../lib/types';
import type { ApiState } from '../App';
import { Card, Section } from './ui';

const OPTIONS: { id: GateBreak; label: string; hint: string }[] = [
  { id: 'none', label: 'Ship as is', hint: 'The committed answer set, exactly what CI checks on every push.' },
  { id: 'approve_seeded', label: 'Let a false pass through', hint: 'A seeded false pass now verifies.' },
  { id: 'drop_known_good', label: 'Break a good answer', hint: 'A correct answer loses its citations.' },
  { id: 'hide_dropped_prompt', label: 'Hide the dropped prompt', hint: 'The dropped system prompt is billed as if it arrived.' },
];

export function ReleaseGate({ api }: { api: ApiState }) {
  const [brk, setBrk] = useState<GateBreak>('none');
  const [run, setRun] = useState<GateRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function go() {
    setBusy(true); setErr(null);
    try { setRun(await runGate(brk)); }
    catch (e) { setRun(null); setErr(e instanceof ApiError ? e.message : 'The gate run failed.'); }
    finally { setBusy(false); }
  }

  return (
    <Section id="gate" step="Step 3 · run it" title="Run the release gate"
      lead={<>This is the gate GitHub Actions runs on every push. It blocks a release if a seeded false pass gets through, if
        known-good answers drop below 6 of 7, or if a dropped system prompt goes unnoticed. Run it as is, or break
        something on purpose and watch it refuse to ship.</>}>
      <Card className="p-5">
        <fieldset>
          <legend className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">Choose what to ship</legend>
          <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {OPTIONS.map((o) => (
              <label key={o.id} className={`cursor-pointer rounded-xl border p-3 transition ${brk === o.id
                ? 'border-emerald-500/50 bg-emerald-500/10' : 'border-zinc-800 bg-zinc-950 hover:border-zinc-600'}`}>
                <input type="radio" name="gate-break" value={o.id} checked={brk === o.id} onChange={() => setBrk(o.id)} className="sr-only" />
                <span className="block text-sm font-medium text-zinc-100">{o.label}</span>
                <span className="mt-0.5 block text-xs leading-relaxed text-zinc-500">{o.hint}</span>
              </label>
            ))}
          </div>
        </fieldset>
        <button type="button" onClick={() => void go()} disabled={busy || api === 'down'}
          className="mt-4 inline-flex items-center gap-2 rounded-xl bg-emerald-500 px-4 py-2.5 text-sm font-semibold text-zinc-950 hover:bg-emerald-400 disabled:opacity-40">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : <Play className="h-4 w-4" aria-hidden />}
          Run the release gate
        </button>
        {api === 'down' && <p className="mt-3 text-sm text-amber-200">The gate runs on the API, which is unreachable right now.</p>}
        {err && <p role="alert" className="mt-3 text-sm text-rose-300">{err}</p>}
      </Card>

      {run && (
        <Card className="mt-4 p-5">
          <div className="flex flex-wrap items-center gap-3">
            <span className={`rounded-lg px-3 py-1.5 font-mono text-sm font-bold ${run.release === 'OK'
              ? 'bg-emerald-500/15 text-emerald-300' : 'bg-rose-500/15 text-rose-300'}`}>
              RELEASE {run.release}
            </span>
            <span className="text-sm text-zinc-400">{run.break_description}</span>
            <span className="ml-auto font-mono text-[11px] text-zinc-500">ran in {run.ran_in_ms} ms on the server</span>
          </div>
          <ul className="mt-4 space-y-2.5">
            {run.checks.map((c) => (
              <li key={c.message} className="flex gap-2.5 text-sm">
                {c.ok ? <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" aria-label="passing" />
                      : <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" aria-label="failing" />}
                <span className={`break-words ${c.ok ? 'text-zinc-300' : 'text-rose-100'}`}>{c.message}</span>
              </li>
            ))}
          </ul>
          <p className="mt-4 text-xs text-zinc-500">{run.note}</p>
        </Card>
      )}
    </Section>
  );
}
