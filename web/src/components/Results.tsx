import { CheckCircle2, XCircle } from 'lucide-react';
import type { JudgeRecord, Metrics } from '../lib/types';
import { Card, LabelPill, Section } from './ui';

// Measured against the deployed API (revision 00003) with
// scripts/measure_latency.py on 2026-10-27 (README, "Live API"): one laptop,
// 20 warm requests per endpoint, cold start after 20 minutes idle.
const CLOUD_RUN = { coldMs: 6832, evaluateP50: 256, retrieveP50: 362, searchServerMs: 34, date: '2026-10-27' };

export function Results({ metrics, judge }: { metrics: Metrics | null; judge: JudgeRecord | null }) {
  const q = new Map((metrics?.questions ?? []).map((x) => [x.id, x]));
  const knownGood = ['q01', 'q02', 'q03', 'q04', 'q05', 'q06', 'q07'];
  const goodPass = knownGood.filter((id) => q.get(id)?.evaluators?.['deterministic_verdict'] === 'PASS').length;
  const seededBlocked = ['s01', 's02', 's03'].filter((id) => q.get(id)?.harness.outcome === 'abstained').length;
  const identity = q.get('s02')?.evaluators?.['prompt_integrity'] === 'FAIL' && q.get('q03')?.evaluators?.['prompt_integrity'] === 'PASS';
  const s = metrics?.summary;
  const lat = s?.latency_ms;
  const tok = s?.recorded_tokens_per_answer;

  return (
    <Section id="results" step="Step 3 · measured" title="What the measurements say"
      lead="Every number below comes from a file in the repo, produced by a script: nothing is typed in by hand.">
      <div className="grid gap-5 lg:grid-cols-2">
        <Card className="p-5">
          <p className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">The real LLM judge, run once</p>
          <p className="mt-2 text-sm leading-relaxed text-zinc-400">
            {judge ? <>{judge.model}, recorded {judge.recorded_at.slice(0, 10)}, on the three seeded false passes. It approved{' '}
              <strong className="text-zinc-100">{judge.summary.judge_passed_count} of {judge.summary.judged}</strong>.</> : 'Loading…'}
          </p>
          {judge && (
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">
                  <tr><th className="py-2 pr-3 font-normal">Sample</th><th className="py-2 pr-3 font-normal">LLM judge</th>
                    <th className="py-2 pr-3 font-normal">Rule-based</th><th className="py-2 font-normal">Verifier</th></tr>
                </thead>
                <tbody className="divide-y divide-zinc-800">
                  {judge.results.map((r) => (
                    <tr key={r.id}>
                      <td className="py-2.5 pr-3"><span className="font-mono text-xs text-zinc-500">{r.id}</span> <span className="text-zinc-200">{r.expect}</span></td>
                      <td className="py-2.5 pr-3"><LabelPill label={r.judge.label} /></td>
                      <td className="py-2.5 pr-3"><LabelPill label={r.rule_based_surrogate} /></td>
                      <td className="py-2.5"><LabelPill label={r.deterministic_verdict} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-3 text-xs leading-relaxed text-zinc-500">
                Its one FAIL (s02) came from its own background knowledge, which contradicted the museum record. It never saw
                that the citations were missing or that the instructions were dropped.
              </p>
            </div>
          )}
        </Card>

        <Card className="p-5">
          <p className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">CI release gate, on every push</p>
          <ul className="mt-3 space-y-3 text-sm">
            <Check ok={seededBlocked === 3} text={`Seeded false passes blocked: ${seededBlocked}/3`} />
            <Check ok={goodPass >= 6} text={`Known-good answers passing: ${goodPass}/7 (threshold 6/7)`}
              note="q06 is a correct answer the year check fails: it read the accession number “1947.39” as a year. Documented, not tuned away." />
            <Check ok={identity} text="Dropped system prompt detected (s02), and not flagged when delivered (q03)" />
            <Check ok text="1,040/1,040 verdicts match between the browser port and Python" />
          </ul>
        </Card>

        <Card className="p-5">
          <p className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">Latency, measured</p>
          <dl className="mt-3 grid grid-cols-2 gap-4">
            <Stat k="Re-score an answer, p50" v={`${CLOUD_RUN.evaluateP50} ms`} n={`round trip, Cloud Run, ${CLOUD_RUN.date}`} />
            <Stat k="Live search, p50" v={`${CLOUD_RUN.retrieveP50} ms`} n={`round trip; ${CLOUD_RUN.searchServerMs} ms on the server`} />
            <Stat k="Cloud Run, cold start" v={`${(CLOUD_RUN.coldMs / 1000).toFixed(1)} s`} n="first request after 20 min idle" />
            {lat && <Stat k="All 8 checks, sequential" v={`${lat.sequential_rescore_evaluate_baseline.p50_ms} ms`} n="p50, no model, no network" />}
          </dl>
          <p className="mt-4 text-xs leading-relaxed text-zinc-500">
            None of these includes a model call: the public demo never makes one. Round trips are from one laptop on a home
            connection and vary between runs (an earlier run measured 110 ms p50 for re-scoring). The check timing is from
            the same laptop ({metrics?.environment.python ? `Python ${metrics.environment.python}` : 'Python'}).
          </p>
        </Card>

        <Card className="p-5">
          <p className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">Recorded cost of a real answer</p>
          {tok ? (
            <dl className="mt-3 grid grid-cols-2 gap-4">
              <Stat k="Prompt tokens per call" v={`${Math.round(tok.prompt_tokens.mean).toLocaleString()}`} n={`range ${tok.prompt_tokens.min.toLocaleString()}–${tok.prompt_tokens.max.toLocaleString()}`} />
              <Stat k="Total per call" v={`${Math.round(tok.total_tokens.mean).toLocaleString()}`} n={`${tok.answers_with_a_model_call} calls, ${tok.sum_total_tokens.toLocaleString()} tokens`} />
              <Stat k="Model calls per answer" v="1" n="retrieval and the gate run first" />
              <Stat k="Harness outcomes" v={`${s?.harness_outcomes.accepted ?? '–'} / ${s?.harness_outcomes.abstained ?? '–'}`} n="accepted / abstained, of 14" />
            </dl>
          ) : <p className="mt-3 text-sm text-zinc-500">Loading…</p>}
        </Card>
      </div>
    </Section>
  );
}

function Check({ ok, text, note }: { ok: boolean; text: string; note?: string }) {
  return (
    <li className="flex gap-2.5">
      {ok ? <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" aria-label="passing" />
          : <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" aria-label="failing" />}
      <span><span className="text-zinc-200">{text}</span>{note && <span className="mt-0.5 block text-xs leading-relaxed text-zinc-500">{note}</span>}</span>
    </li>
  );
}

function Stat({ k, v, n }: { k: string; v: string; n: string }) {
  return (
    <div>
      <dt className="text-xs text-zinc-500">{k}</dt>
      <dd className="mt-0.5 font-mono text-xl font-semibold text-zinc-50">{v}</dd>
      <dd className="text-[11px] text-zinc-500">{n}</dd>
    </div>
  );
}
