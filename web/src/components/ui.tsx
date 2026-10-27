import type { ReactNode } from 'react';
import type { Label } from '../lib/types';

export function Section({ id, step, title, lead, children }: {
  id: string; step?: string; title: string; lead?: ReactNode; children: ReactNode;
}) {
  return (
    <section id={id} className="mx-auto max-w-6xl px-4 sm:px-6 py-14 sm:py-20">
      <div className="max-w-3xl">
        {step && <p className="font-mono text-xs uppercase tracking-widest text-emerald-400/90">{step}</p>}
        <h2 className="mt-2 text-2xl sm:text-3xl font-semibold tracking-tight text-zinc-50">{title}</h2>
        {lead && <p className="mt-3 text-zinc-400 leading-relaxed">{lead}</p>}
      </div>
      <div className="mt-8">{children}</div>
    </section>
  );
}

export function Card({ className = '', children }: { className?: string; children: ReactNode }) {
  return <div className={`rounded-2xl border border-zinc-800 bg-zinc-900/60 ${className}`}>{children}</div>;
}

const TONES: Record<Label, string> = {
  PASS: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
  FAIL: 'border-rose-500/40 bg-rose-500/10 text-rose-300',
  INCONCLUSIVE: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
  NOT_APPLICABLE: 'border-zinc-600 bg-zinc-800/60 text-zinc-400',
};

const SHORT: Record<Label, string> = { PASS: 'PASS', FAIL: 'FAIL', INCONCLUSIVE: 'INCONCLUSIVE', NOT_APPLICABLE: 'N/A' };

export function LabelPill({ label, big = false }: { label: Label; big?: boolean }) {
  return (
    <span className={`inline-flex items-center rounded-full border font-mono font-semibold ${TONES[label]}
      ${big ? 'px-3 py-1 text-sm' : 'px-2 py-0.5 text-[11px]'}`}>
      {SHORT[label]}
    </span>
  );
}

export function Tag({ children, tone = 'zinc' }: { children: ReactNode; tone?: 'zinc' | 'emerald' | 'sky' | 'amber' }) {
  const t = {
    zinc: 'border-zinc-700 text-zinc-400',
    emerald: 'border-emerald-500/40 text-emerald-300',
    sky: 'border-sky-500/40 text-sky-300',
    amber: 'border-amber-500/40 text-amber-300',
  }[tone];
  return <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-[11px] ${t}`}>{children}</span>;
}

export function ExtLink({ href, children, className = '' }: { href: string; children: ReactNode; className?: string }) {
  return <a href={href} target="_blank" rel="noopener noreferrer" className={className}>{children}</a>;
}
