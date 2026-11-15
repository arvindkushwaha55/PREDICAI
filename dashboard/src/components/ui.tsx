import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { PolicyLevel } from "../lib/data";

export function Card({
  children,
  className = "",
  ticks = false,
}: {
  children: ReactNode;
  className?: string;
  /** Bracket accents on two corners (engineering-console framing). */
  ticks?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-sm transition-colors duration-200 hover:border-slate-300 dark:hover:border-slate-600 ${ticks ? "corner-ticks" : ""} ${className}`}
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
      <h2 className="flex items-center gap-2 font-mono text-[13px] font-semibold tracking-[0.14em] text-slate-900 uppercase dark:text-slate-100">
        <span className="inline-block h-2 w-2 shrink-0 bg-cyan-600 dark:bg-cyan-400" />
        {title}
      </h2>
      {source && <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{source}</p>}
    </div>
  );
}

/** Small source label shown on panels, e.g. "FD001 REPLAY". */
export function SourceTag({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center rounded border border-cyan-700/30 bg-cyan-50 dark:bg-cyan-950 px-1.5 py-0.5 text-[11px] font-semibold tracking-wide text-cyan-800 dark:text-cyan-300">
      {label}
    </span>
  );
}

const LEVEL_STYLES: Record<PolicyLevel, string> = {
  MONITOR: "bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300 border-emerald-600/30",
  INSPECTION: "bg-amber-100 dark:bg-amber-950 text-amber-900 dark:text-amber-200 border-amber-600/40",
  REVIEW: "bg-rose-100 dark:bg-rose-950 text-rose-800 dark:text-rose-300 border-rose-600/30",
};

export function LevelChip({ level }: { level: PolicyLevel | string }) {
  const style =
    LEVEL_STYLES[level as PolicyLevel] ??
    "bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-400/40";
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
  value: ReactNode;
  sub?: string;
}) {
  return (
    <div>
      <div className="font-mono text-[11px] font-medium tracking-[0.14em] text-slate-500 uppercase dark:text-slate-400">
        {label}
      </div>
      <div className="tabular mono mt-1 text-xl font-semibold text-slate-900 dark:text-slate-100">
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{sub}</div>}
    </div>
  );
}

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

/**
 * Counts from 0 up to `value` once, on first mount (ease-out, under a
 * second), then lands exactly on `value`. Presentation only: the number
 * displayed at rest is always the real value, formatted by `format`.
 * Later value changes apply immediately, with no re-animation. With
 * prefers-reduced-motion the value is shown directly.
 */
export function CountUp({
  value,
  format,
  duration = 700,
}: {
  value: number;
  format: (n: number) => string;
  duration?: number;
}) {
  const [display, setDisplay] = useState<number>(() =>
    prefersReducedMotion() ? value : 0,
  );
  const animated = useRef(false);

  useEffect(() => {
    if (animated.current) {
      setDisplay(value);
      return;
    }
    animated.current = true;
    if (prefersReducedMotion()) {
      setDisplay(value);
      return;
    }
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setDisplay(value * eased);
      if (t < 1) {
        raf = requestAnimationFrame(tick);
      } else {
        setDisplay(value);
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);

  return <>{format(display)}</>;
}
