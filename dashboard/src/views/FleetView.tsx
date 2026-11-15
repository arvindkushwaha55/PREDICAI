import { data, fmt, policyForRul } from "../lib/data";
import type { EngineData } from "../lib/types";
import { Card, CountUp, LevelChip, SectionTitle, SourceTag, Stat } from "../components/ui";

function EngineCard({
  engine,
  onOpen,
}: {
  engine: EngineData;
  onOpen: (engine: number) => void;
}) {
  const policy = policyForRul(engine.final.predicted_rul);
  return (
    <Card className="flex flex-col p-5" ticks>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-lg font-bold break-words text-slate-900 dark:text-slate-100">
            {engine.machine_id}
          </div>
          <div className="text-xs text-slate-500 dark:text-slate-400">
            FD001 test engine {engine.engine} · {engine.total_recorded_cycles}{" "}
            recorded cycles
          </div>
        </div>
        <SourceTag label="FD001 REPLAY" />
      </div>

      <div className="tabular mt-4 rounded border border-slate-200 bg-slate-50 px-3 py-2.5 dark:border-slate-700 dark:bg-slate-800">
        <div className="font-mono text-[11px] font-medium tracking-[0.14em] text-slate-500 uppercase dark:text-slate-400">
          Final predicted RUL
        </div>
        <div className="mono mt-0.5 text-3xl font-bold text-cyan-800 sm:text-4xl dark:text-cyan-300">
          <CountUp value={engine.final.predicted_rul} format={fmt} />{" "}
          <span className="text-base font-semibold text-slate-500 dark:text-slate-400">cycles</span>
        </div>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Stat
          label="True RUL (final)"
          value={`${fmt(engine.final.true_rul, 0)} cycles`}
        />
        <Stat
          label="Final abs. error"
          value={`${fmt(engine.final.abs_error)} cycles`}
        />
        <Stat
          label="Predictions logged"
          value={String(engine.predictions_logged)}
        />
        <Stat
          label="Replay MAE vs true"
          value={
            engine.stats.mean_abs_error_vs_true != null
              ? `${fmt(engine.stats.mean_abs_error_vs_true)} cycles`
              : "-"
          }
        />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-slate-100 dark:border-slate-800 pt-4">
        <LevelChip level={policy.level} />
        <span className="text-sm text-slate-600 dark:text-slate-400">{policy.action}</span>
      </div>
      <p className="mt-1 text-xs text-slate-400 dark:text-slate-500">
        Status from the policy RUL band only (policy.py v1.0 thresholds).
      </p>

      <button
        onClick={() => onOpen(engine.engine)}
        className="mt-4 rounded border border-slate-300 dark:border-slate-600 px-3 py-2 text-sm font-semibold text-slate-700 dark:text-slate-300 hover:border-cyan-700 hover:text-cyan-800 dark:text-cyan-300"
      >
        Open machine detail →
      </button>
    </Card>
  );
}

export function FleetView({ onOpen }: { onOpen: (engine: number) => void }) {
  const t = data.policy_thresholds;
  return (
    <div className="stagger-cards space-y-6">
      <div className="stagger-cards grid gap-4 md:grid-cols-3">
        {data.engines.map((e) => (
          <EngineCard key={e.machine_id} engine={e} onOpen={onOpen} />
        ))}
      </div>

      <Card className="p-5" ticks>
        <SectionTitle
          title="How status is decided: policy thresholds"
          source={t.source}
        />
        <div className="tabular grid gap-3 text-sm text-slate-700 dark:text-slate-300 md:grid-cols-3">
          <div className="rounded border border-rose-600/20 bg-rose-50 dark:bg-rose-950 px-3 py-2">
            <strong>REVIEW</strong>: predicted RUL ≤ {t.review_max_rul} cycles
          </div>
          <div className="rounded border border-amber-600/30 bg-amber-50 dark:bg-amber-950 px-3 py-2">
            <strong>INSPECTION</strong>: predicted RUL ≤{" "}
            {t.inspection_max_rul} cycles
          </div>
          <div className="rounded border border-emerald-600/20 bg-emerald-50 dark:bg-emerald-950 px-3 py-2">
            <strong>MONITOR</strong>: predicted RUL &gt;{" "}
            {t.inspection_max_rul} cycles
          </div>
        </div>
        <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">
          The full policy also escalates a level when a machine has gone ≥{" "}
          {t.service_review_cycles} cycles since its last service. Service
          history exists only for the synthetic demo machine (Maintenance
          view), so fleet status above uses the RUL band alone. A status is a
          planning prompt, never a claim that a machine will fail at a
          specific cycle.
        </p>
      </Card>
    </div>
  );
}
