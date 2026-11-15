import { platform } from "../lib/platformData";
import type { PlatformDataset } from "../lib/platformTypes";
import { Card, SectionTitle } from "../components/ui";

const STATUS_STYLES: Record<PlatformDataset["status"], string> = {
  integrated: "bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300 border-emerald-600/30",
  blocked: "bg-rose-100 dark:bg-rose-950 text-rose-800 dark:text-rose-300 border-rose-600/30",
  researched: "bg-amber-100 dark:bg-amber-950 text-amber-900 dark:text-amber-200 border-amber-600/40",
  "not-found": "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 border-slate-400/40",
};

const STATUS_LABELS: Record<PlatformDataset["status"], string> = {
  integrated: "INTEGRATED",
  blocked: "BLOCKED",
  researched: "RESEARCHED ONLY",
  "not-found": "NO VERIFIED DATASET",
};

const TASK_NAMES: Record<string, string> = {
  A: "A · Classification",
  B: "B · RUL",
  C: "C · Anomaly",
};

function StatusChip({ status }: { status: PlatformDataset["status"] }) {
  return (
    <span
      className={`inline-flex items-center rounded border px-2 py-0.5 text-xs font-bold tracking-wide ${STATUS_STYLES[status]}`}
    >
      {STATUS_LABELS[status]}
    </span>
  );
}

export function DatasetsView() {
  const ds = platform.datasets;
  const nBlocked = ds.filter((d) => d.status === "blocked").length;
  const nResearched = ds.filter((d) => d.status === "researched").length;
  return (
    <div className="stagger-cards space-y-6">
      <Card className="p-5" ticks>
        <SectionTitle
          title="Dataset registry"
          source="Facts transcribed from platform/DATASET_REGISTRY.md (verified 2026-11-15). Blocked and researched-only datasets are listed on purpose; a registry that only shows successes would misrepresent the platform's coverage."
        />
        <p className="text-sm text-slate-600 dark:text-slate-400">
          {platform.counts.datasets_integrated} integrated · {nBlocked}{" "}
          blocked · {nResearched} researched-only ·{" "}
          {ds.length - platform.counts.datasets_integrated - nBlocked - nResearched}{" "}
          category with no verified dataset. Licence text quotes what the
          official source states; “not stated / not confirmed” does not
          mean public domain.
        </p>
      </Card>

      <div className="stagger-cards grid gap-4 lg:grid-cols-2">
        {ds.map((d) => (
          <Card
            key={d.id}
            className={
              d.status === "integrated"
                ? "p-5"
                : "border-dashed bg-slate-50 dark:bg-slate-800 p-5 opacity-90"
            }
          >
            <div className="flex flex-wrap items-start justify-between gap-2">
              <h3 className="text-base font-semibold text-slate-900 dark:text-slate-100">
                {d.name}
              </h3>
              <StatusChip status={d.status} />
            </div>
            <dl className="mt-3 space-y-1.5 text-sm">
              <div className="flex gap-2">
                <dt className="w-20 shrink-0 text-xs font-medium tracking-wide text-slate-500 dark:text-slate-400 uppercase pt-0.5">
                  Source
                </dt>
                <dd className="text-slate-700 dark:text-slate-300">{d.source}</dd>
              </div>
              <div className="flex gap-2">
                <dt className="w-20 shrink-0 text-xs font-medium tracking-wide text-slate-500 dark:text-slate-400 uppercase pt-0.5">
                  Licence
                </dt>
                <dd className="text-slate-700 dark:text-slate-300">{d.licence}</dd>
              </div>
              <div className="flex gap-2">
                <dt className="w-20 shrink-0 text-xs font-medium tracking-wide text-slate-500 dark:text-slate-400 uppercase pt-0.5">
                  Tasks
                </dt>
                <dd className="flex flex-wrap gap-1">
                  {d.tasks.length > 0 ? (
                    d.tasks.map((t) => (
                      <span
                        key={t}
                        className="inline-flex items-center rounded border border-cyan-700/30 bg-cyan-50 dark:bg-cyan-950 px-1.5 py-0.5 text-[11px] font-semibold text-cyan-800 dark:text-cyan-300"
                      >
                        {TASK_NAMES[t] ?? t}
                      </span>
                    ))
                  ) : (
                    <span className="text-slate-500 dark:text-slate-400">-</span>
                  )}
                </dd>
              </div>
              <div className="flex gap-2">
                <dt className="w-20 shrink-0 text-xs font-medium tracking-wide text-slate-500 dark:text-slate-400 uppercase pt-0.5">
                  Adapter
                </dt>
                <dd className="text-slate-700 dark:text-slate-300">
                  {d.adapter ?? "-"}
                  {d.records > 0 && (
                    <span className="text-slate-500 dark:text-slate-400">
                      {" "}
                      · {d.records} experiment record
                      {d.records === 1 ? "" : "s"}
                    </span>
                  )}
                </dd>
              </div>
            </dl>
                {d.specs.length > 0 && (
                  <div className="mt-3 grid grid-cols-2 gap-px overflow-hidden rounded border border-slate-200 bg-slate-200 dark:border-slate-700 dark:bg-slate-700 sm:grid-cols-4">
                    {d.specs.map(([k, v]) => (
                      <div key={k} className="bg-slate-50 px-2.5 py-2 dark:bg-slate-800">
                        <div className="font-mono text-[10px] font-medium tracking-wider text-slate-500 uppercase dark:text-slate-400">
                          {k}
                        </div>
                        <div className="mono mt-0.5 text-[13px] font-semibold text-slate-800 dark:text-slate-200">
                          {v}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
            <p className="mt-3 border-t border-slate-100 dark:border-slate-800 pt-3 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
              {d.note}
            </p>
          </Card>
        ))}
      </div>
    </div>
  );
}
