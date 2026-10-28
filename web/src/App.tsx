import { useCallback, useEffect, useState } from 'react';
import { FoolTheVerifier } from './components/FoolTheVerifier';
import { Header } from './components/Header';
import { Hero } from './components/Hero';
import { HowItWorks } from './components/HowItWorks';
import { ReleaseGate } from './components/ReleaseGate';
import { Results } from './components/Results';
import { SearchArchive } from './components/SearchArchive';
import { ExtLink } from './components/ui';
import { health } from './lib/api';
import { LINKS, loadJudge, loadMetrics, loadRecorded } from './lib/data';
import type { JudgeRecord, Metrics, Sample } from './lib/types';

export type ApiState = 'checking' | 'up' | 'down';

export default function App() {
  const [api, setApi] = useState<ApiState>('checking');
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selected, setSelected] = useState('s01');
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [judge, setJudge] = useState<JudgeRecord | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    health().then((h) => setApi(h.model_calls === 'disabled' ? 'up' : 'down')).catch(() => setApi('down'));
    loadRecorded().then((d) => setSamples(d.scenarios)).catch((e: unknown) => setLoadError(String(e)));
    loadMetrics().then(setMetrics).catch(() => setMetrics(null));
    loadJudge().then(setJudge).catch(() => setJudge(null));
  }, []);

  const openSample = useCallback((id: string) => {
    setSelected(id);
    document.getElementById('verify')?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  return (
    <>
      <Header api={api} />
      <main>
        <Hero />
        <SearchArchive api={api} onOpenSample={openSample} />
        {loadError ? (
          <p role="alert" className="mx-auto max-w-6xl px-4 sm:px-6 text-rose-300">Could not load the recorded samples: {loadError}</p>
        ) : samples.length > 0 && (
          <FoolTheVerifier key={selected} api={api} samples={samples} selected={selected}
            onSelect={setSelected} onApiFailed={() => setApi('down')} />
        )}
        <ReleaseGate api={api} />
        <Results metrics={metrics} judge={judge} />
        <HowItWorks />
      </main>
      <footer className="border-t border-zinc-800">
        <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 sm:px-6 py-8 text-sm text-zinc-500 sm:flex-row sm:items-center">
          <p>Built by <span className="text-zinc-300">Gargi Vipat</span>. Records: Cleveland Museum of Art (CC0) and Art Institute of Chicago (CC0; descriptions CC BY 4.0).</p>
          <div className="flex gap-4 sm:ml-auto">
            <ExtLink href={LINKS.repo} className="hover:text-zinc-200">GitHub</ExtLink>
            <ExtLink href={LINKS.linkedin} className="hover:text-zinc-200">LinkedIn</ExtLink>
            <ExtLink href={LINKS.portfolio} className="hover:text-zinc-200">Portfolio</ExtLink>
          </div>
        </div>
      </footer>
    </>
  );
}
