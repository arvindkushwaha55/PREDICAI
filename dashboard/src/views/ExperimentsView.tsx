import { useMemo, useState } from "react";
import {
  platform,
  datasetLabel,
  fmtMetric,
  metricLabel,
  fmtTrainTime,
} from "../lib/platformData";
import type { RecordKind } from "../lib/platformTypes";
import { Card, SectionTitle } from "../components/ui";

const KIND_STYLES: Record<RecordKind, string> = {
  experiment: "bg-slate-100 text-slate-600 border-slate-400/40",
  transfer: "bg-violet-100 text-violet-800 border-violet-600/30",
  explainability: "bg-cyan-50 text-cyan-800 border-cyan-700/30",
};

const KIND_LABELS: Record<RecordKind, string> = {
  experiment: "TRAINED",
  transfer: "TRANSFER",
  explainability: "XAI",
};

function HeadlineCell({ headline }: { headline: Record<string, number | string> }) {
  const entries = Object.entries(headline);
  if (entries.length === 0) return <span className="text-slate-400">—</span>;
  return (
    <span className="space-x-3">
      {entries.map(([k, v]) => (
        <span key={k} className="whitespace-nowrap">
          <span className="text-slate-500">{metricLabel(k)} </span>
          <span className="font-semibold text-slate-800">
            {typeof v === "number" ? fmtMetric(v) : v}
          </span>
        </span>
      ))}
    </span>
  );
}

export function ExperimentsView() {
  const [dsFilter, setDsFilter] = useState<string>("all");
  const [taskFilter, setTaskFilter] = useState<string>("all");

  const datasetIds = useMemo(() => {
    const seen = new Map<string, string>();
    for (const e of platform.experiments) {
      if (!seen.has(e.dataset)) seen.set(e.dataset, datasetLabel(e.dataset));
    }
    return [...seen.entries()].sort((a, b) => a[1].localeCompare(b[1]));
  }, []);

  const rows = platform.experiments.filter(
    (e) =>
      (dsFilter === "all" || e.dataset === dsFilter) &&
      (taskFilter === "all" || e.task === taskFilter),
  );

  const selectCls =
    "rounded border border-slate-300 bg-white px-2 py-1.5 text-sm text-slate-700";

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <SectionTitle
          title={`Experiment registry — ${rows.length} of ${platform.counts.records} records`}
          source="One row per record JSON in platform/experiments/. Headline metrics are the records' own curated headlines; the protocol column states the split each score was measured under — scores under different protocols are not interchangeable."
        />
        <div className="mb-4 flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-slate-600">
            Dataset
            <select
              value={dsFilter}
              onChange={(e) => setDsFilter(e.target.value)}
              className={selectCls}
            >
              <option value="all">All datasets</option>
              {datasetIds.map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 text-sm text-slate-600">
            Task
            <select
              value={taskFilter}
              onChange={(e) => setTaskFilter(e.target.value)}
              className={selectCls}
            >
              <option value="all">All tasks</option>
              <option value="A">A — Classification</option>
              <option value="B">B — RUL regression</option>
              <option value="C">C — Anomaly detection</option>
            </select>
          </label>
          {(dsFilter !== "all" || taskFilter !== "all") && (
            <button
              onClick={() => {
                setDsFilter("all");
                setTaskFilter("all");
              }}
              className="text-sm font-medium text-cyan-700 hover:underline"
            >
              Clear filters
            </button>
          )}
        </div>

        <div className="overflow-x-auto">
          <table className="tabular w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-xs tracking-wide text-slate-500 uppercase">
                <th className="py-2 pr-4">Record</th>
                <th className="py-2 pr-4">Task</th>
                <th className="py-2 pr-4">Headline metrics</th>
                <th className="py-2 pr-4">Train time</th>
                <th className="py-2">Split / protocol</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((e) => (
                <tr key={e.id} className="border-b border-slate-100 align-top">
                  <td className="py-2 pr-4">
                    <div className="font-medium text-slate-800">
                      {datasetLabel(e.dataset)}
                    </div>
                    <div className="text-xs text-slate-500">
                      {e.model} · seed {e.seed ?? "—"}
                    </div>
                    <span
                      className={`mt-1 inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-bold tracking-wide ${KIND_STYLES[e.kind]}`}
                    >
                      {KIND_LABELS[e.kind]}
                    </span>
                  </td>
                  <td className="py-2 pr-4 font-semibold text-slate-700">
                    {e.task}
                  </td>
                  <td className="py-2 pr-4">
                    <HeadlineCell headline={e.headline} />
                  </td>
                  <td className="py-2 pr-4 whitespace-nowrap">
                    {fmtTrainTime(e.train_s)}
                  </td>
                  <td className="py-2 text-xs leading-relaxed text-slate-500">
                    <span className="line-clamp-3" title={e.protocol ?? ""}>
                      {e.protocol ?? "—"}
                    </span>
                  </td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={5} className="py-6 text-center text-slate-500">
                    No records match these filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-slate-500">
          Full per-record detail (hyperparameters, environment, confusion
          matrices, per-engine predictions, caveats) lives in the record
          JSONs themselves — this table is an index, not a substitute.
        </p>
      </Card>
    </div>
  );
}
