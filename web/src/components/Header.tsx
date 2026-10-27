import { Github, Globe, Linkedin, ShieldCheck } from 'lucide-react';
import { LINKS } from '../lib/data';
import type { ApiState } from '../App';
import { ExtLink } from './ui';

const STATUS: Record<ApiState, { dot: string; text: string; title: string }> = {
  checking: { dot: 'bg-zinc-500 animate-pulse', text: 'Waking the API…',
              title: 'The Cloud Run API scales to zero when idle; the first request starts it (measured cold start: 6.8 s).' },
  up: { dot: 'bg-emerald-400', text: 'Live API', title: 'Cloud Run API is up: evaluators run in Python, search runs live.' },
  down: { dot: 'bg-amber-400', text: 'Offline mode',
          title: 'The API is unreachable. Verdicts are computed in your browser by the parity-checked TypeScript port; live search is unavailable.' },
};

export function Header({ api }: { api: ApiState }) {
  const s = STATUS[api];
  return (
    <header className="sticky top-0 z-30 border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 sm:px-6 h-14">
        <a href="#top" className="flex items-center gap-2 min-w-0">
          <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-emerald-500/30 bg-emerald-500/10">
            <ShieldCheck className="h-4 w-4 text-emerald-400" aria-hidden />
          </span>
          <span className="truncate font-semibold text-zinc-100">Museum QA Verifiers</span>
        </a>
        <nav className="ml-6 hidden md:flex items-center gap-5 text-sm text-zinc-400" aria-label="Sections">
          <a className="hover:text-zinc-100" href="#search">Search</a>
          <a className="hover:text-zinc-100" href="#verify">Fool the verifier</a>
          <a className="hover:text-zinc-100" href="#results">Results</a>
          <a className="hover:text-zinc-100" href="#how">How it works</a>
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <span title={s.title} role="status" className="inline-flex items-center gap-2 rounded-full border border-zinc-800 px-2.5 py-1 font-mono text-[11px] text-zinc-400">
            <span className={`h-2 w-2 rounded-full ${s.dot}`} aria-hidden /><span className="hidden sm:inline">{s.text}</span>
            <span className="sr-only sm:hidden">{s.text}</span>
          </span>
          <ExtLink href={LINKS.repo} className="rounded-lg border border-zinc-800 p-2 text-zinc-400 hover:text-zinc-100 hover:border-zinc-600">
            <Github className="h-4 w-4" aria-label="GitHub repository" />
          </ExtLink>
          <ExtLink href={LINKS.linkedin} className="rounded-lg border border-zinc-800 p-2 text-zinc-400 hover:text-zinc-100 hover:border-zinc-600">
            <Linkedin className="h-4 w-4" aria-label="LinkedIn" />
          </ExtLink>
          <ExtLink href={LINKS.portfolio} className="rounded-lg border border-zinc-800 p-2 text-zinc-400 hover:text-zinc-100 hover:border-zinc-600">
            <Globe className="h-4 w-4" aria-label="Portfolio" />
          </ExtLink>
        </div>
      </div>
    </header>
  );
}
