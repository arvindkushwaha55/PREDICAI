import type { ReactNode } from "react";
import type { PolicyLevel } from "../lib/data";

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-lg border border-slate-200 bg-white shadow-sm ${className}`}
    >
      {children}
    </div>
  );
}

export function SectionTitle({
  title,
  source,
}: {
  title: string;
  source?: string;
}) {
  return (
    <div className="mb-3">
      <h2 className="text-base font-semibold text-slate-900">{title}</h2>
      {source && <p className="mt-0.5 text-xs text-slate-500">{source}</p>}
    </div>
  );
}

/** Small source label shown on panels, e.g. "FD001 REPLAY". */
export function SourceTag({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center rounded border border-cyan-700/30 bg-cyan-50 px-1.5 py-0.5 text-[11px] font-semibold tracking-wide text-cyan-800">
      {label}
    </span>
  );
}

const LEVEL_STYLES: Record<PolicyLevel, string> = {
  MONITOR: "bg-emerald-100 text-emerald-800 border-emerald-600/30",
  INSPECTION: "bg-amber-100 text-amber-900 border-amber-600/40",
  REVIEW: "bg-rose-100 text-rose-800 border-rose-600/30",
};

export function LevelChip({ level }: { level: PolicyLevel | string }) {
  const style =
    LEVEL_STYLES[level as PolicyLevel] ??
    "bg-slate-100 text-slate-700 border-slate-400/40";
  return (
    <span
      className={`inline-flex items-center rounded border px-2 py-0.5 text-xs font-bold tracking-wide ${style}`}
    >
      {level}
    </span>
  );
}

export function Stat({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div>
      <div className="text-xs font-medium tracking-wide text-slate-500 uppercase">
        {label}
      </div>
      <div className="tabular mt-1 text-xl font-semibold text-slate-900">
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}
