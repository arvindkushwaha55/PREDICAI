import { data, fmt } from "../lib/data";
import {
  platform,
  fmtMetric,
  transferPairLabel,
} from "../lib/platformData";
import type { RefMetrics, TransferRecord } from "../lib/platformTypes";
import { Card, SectionTitle } from "../components/ui";

function metricsText(m: RefMetrics | TransferRecord["cross_metrics"]): string {
  const parts: string[] = [];
  if (m.rmse != null) parts.push(`RMSE ${fmtMetric(m.rmse)}`);
  if (m.mae != null) parts.push(`MAE ${fmtMetric(m.mae)}`);
  if (m.nasa_score != null) parts.push(`NASA ${fmtMetric(m.nasa_score)}`);
  if (m.accuracy != null) parts.push(`Acc ${fmtMetric(m.accuracy)}`);
  if (m.f1_macro != null) parts.push(`F1-macro ${fmtMetric(m.f1_macro)}`);
  return parts.join(" · ") || "-";
}

function TransfersCard() {
  return (
    <Card className="p-5" ticks>
      <SectionTitle
        title="Cross-dataset generalisation: zero-shot transfer"
        source="Each row: a model trained on the source dataset applied to the target with NO target fitting (source scaler, source features). In-domain references are the same target scored by a model trained on it. Negative results are reported as measured; they are the finding, not a failure of the table."
      />
      <div className="overflow-x-auto">
        <table className="tabular mono w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-700 text-xs tracking-wide text-slate-500 dark:text-slate-400 uppercase">
              <th className="py-2 pr-4">Transfer</th>
              <th className="py-2 pr-4">Model</th>
              <th className="py-2 pr-4">Cross-domain</th>
              <th className="py-2 pr-4">In-domain reference(s)</th>
              <th className="py-2">Verdict (from the record)</th>
            </tr>
          </thead>
          <tbody>
            {platform.transfers.map((t) => (
              <tr key={t.id} className="border-b border-slate-100 dark:border-slate-800 align-top transition-colors duration-150 hover:bg-slate-50 dark:hover:bg-slate-800">
                <td className="py-2 pr-4 font-medium whitespace-nowrap text-slate-800 dark:text-slate-200">
                  {transferPairLabel(t.source_dataset, t.target_dataset, t.id)}
                </td>
                <td className="py-2 pr-4 whitespace-nowrap">{t.model}</td>
                <td className="py-2 pr-4 font-semibold whitespace-nowrap">
                  {metricsText(t.cross_metrics)}
                </td>
                <td className="py-2 pr-4 text-slate-600 dark:text-slate-400">
                  {t.in_domain_references.map((r) => (
                    <div key={r.label} className="whitespace-nowrap">
                      <span className="text-slate-500 dark:text-slate-400">{r.label}: </span>
                      {metricsText(r)}
                    </div>
                  ))}
                </td>
                <td className="py-2 text-slate-600 dark:text-slate-400">{t.verdict ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
        Pattern across all {platform.transfers.length} records: transfer
        works when the source domain is at least as wide as the target
        (FD003→FD001, FD004→FD002) and fails when the target contains
        structure the source never saw (FD001→FD003's unseen fan fault
        mode; a different bearing rig's “Normal”). Feature alignment
        alone does not fix distribution shift; Wave 9 tested that hope
        directly: adding spectral + envelope features moved the bearing
        transfers only marginally (one direction slightly worse, one
        slightly better) and both still fail. Wave 12 tested a third
        family: order-domain features, which align the characteristic
        defect frequencies in shaft-speed-relative units: in-domain
        CWRU diagnosis jumped to 0.802, yet both transfers still fail
        and Normal F1 is 0.000 again. With hand-crafted features, the
        cross-rig question is closed negatively.
      </p>
    </Card>
  );
}

function XaiBars({
  rows,
}: {
  rows: { feature: string; value: number | null }[];
}) {
  const vals = rows.map((r) => r.value ?? 0);
  const max = Math.max(...vals, 0);
  return (
    <div className="space-y-1.5">
      {rows.map((r) => (
        <div key={r.feature} className="flex items-center gap-2 text-sm">
          <span className="w-44 shrink-0 truncate text-slate-700 dark:text-slate-300" title={r.feature}>
            {r.feature}
          </span>
          <span className="h-2.5 flex-1 rounded bg-slate-100 dark:bg-slate-800">
            <span
              className="block h-2.5 rounded bg-cyan-600"
              style={{
                width: `${max > 0 ? Math.max(1.5, ((r.value ?? 0) / max) * 100) : 0}%`,
              }}
            />
          </span>
          <span className="tabular w-14 shrink-0 text-right font-medium text-slate-800 dark:text-slate-200">
            {r.value != null ? fmtMetric(r.value) : "-"}
          </span>
        </div>
      ))}
    </div>
  );
}

function XaiCard() {
  const fd = data.model.explainability;
  return (
    <Card className="p-5">
      <SectionTitle
        title="Explainability highlights"
        source="Rankings are read from the explainability records (SHAP / permutation) and from the shipped model's Part 6 explainability export. Attributions describe what a model relies on (association, not physical causation)."
      />
      <div className="grid gap-6 lg:grid-cols-2">
        {platform.xai.map((x) => (
          <div key={x.id}>
            <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
              {x.method}: {x.dataset} ({x.model})
            </h3>
            <p className="mb-2 font-mono text-[11px] text-slate-400 dark:text-slate-500">{x.id}</p>
            <XaiBars rows={x.ranking.slice(0, 5)} />
            {x.notes && (
              <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
                {x.notes}
              </p>
            )}
          </div>
        ))}

        <div>
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Permutation + Integrated Gradients: shipped FD001 LSTM
          </h3>
          <p className="mb-2 text-[11px] text-slate-400 dark:text-slate-500">
            From demo_data.json (ml/artifacts/explainability.json),{" "}
            {fd.split}, n = {fd.n_val_windows.toLocaleString("en-US")}{" "}
            windows. Also shown in full in Model Info.
          </p>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <div className="mb-1 text-xs font-medium tracking-wide text-slate-500 dark:text-slate-400 uppercase">
                Permutation: RMSE increase
              </div>
              <XaiBars
                rows={fd.permutation_top5.map((r) => ({
                  feature: r.sensor,
                  value: r.rmse_increase,
                }))}
              />
            </div>
            <div>
              <div className="mb-1 text-xs font-medium tracking-wide text-slate-500 dark:text-slate-400 uppercase">
                Integrated Gradients: |attribution|
              </div>
              <XaiBars
                rows={fd.integrated_gradients_top5.map((r) => ({
                  feature: r.sensor,
                  value: r.mean_abs_attribution,
                }))}
              />
            </div>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
            Top-5 overlap between the two methods:{" "}
            {fd.top5_overlap.join(", ") || "none"}. The rankings otherwise
            differ; expected, since the methods measure different things
            (accuracy reliance vs per-prediction attribution).
          </p>
        </div>
      </div>
    </Card>
  );
}

function ImsCard() {
  const ims = platform.ims_alarms;
  if (!ims) return null;
  return (
    <Card className="p-5">
      <SectionTitle
        title="IMS run-to-failure: anomaly alarm timing (Test 1)"
        source={`${ims.experiment_id} · ${ims.n_snapshots.toLocaleString("en-US")} snapshots, one Isolation Forest per bearing fitted ONLY on that bearing's first 30% of snapshots (early life). ${ims.framing}`}
      />
      <div className="overflow-x-auto">
        <table className="tabular mono w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200 dark:border-slate-700 text-xs tracking-wide text-slate-500 dark:text-slate-400 uppercase">
              <th className="py-2 pr-4">Bearing</th>
              <th className="py-2 pr-4">Teardown ground truth</th>
              <th className="py-2 pr-4">First sustained alarm</th>
              <th className="py-2 pr-4">Warning before run end</th>
              <th className="py-2">Last-decile exceedance</th>
            </tr>
          </thead>
          <tbody>
            {ims.bearings.map((b) => (
              <tr key={b.bearing} className="border-b border-slate-100 dark:border-slate-800 transition-colors duration-150 hover:bg-slate-50 dark:hover:bg-slate-800">
                <td className="py-2 pr-4 font-medium text-slate-800 dark:text-slate-200 capitalize">
                  {b.bearing}
                </td>
                <td className="py-2 pr-4">
                  {b.documented_defect ?? (
                    <span className="text-slate-500 dark:text-slate-400">no defect reported</span>
                  )}
                </td>
                <td className="py-2 pr-4">
                  {b.first_sustained_alarm_index != null ? (
                    <>
                      snapshot{" "}
                      {b.first_sustained_alarm_index.toLocaleString("en-US")}{" "}
                      <span className="text-slate-500 dark:text-slate-400">
                        ({b.first_sustained_alarm_timestamp})
                      </span>
                    </>
                  ) : (
                    <span className="text-slate-500 dark:text-slate-400">none</span>
                  )}
                </td>
                <td className="py-2 pr-4 font-semibold">
                  {b.snapshots_of_warning_before_run_end > 0
                    ? `${b.snapshots_of_warning_before_run_end.toLocaleString("en-US")} snapshots`
                    : "-"}
                </td>
                <td className="py-2">
                  {fmt(b.last_decile_exceedance_rate * 100, 1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
        The honest success pattern for this design: sustained alarms on
        exactly the two bearings that failed (3 and 4, 337 and 723
        snapshots of warning), and none on the two that did not (1 and 2).
        That is what was measured for Test 1. Tests 2–3 (Wave 9, same
        protocol, no re-tuning) complicate the picture: in Test 2 the
        failed bearing (1, outer race) alarmed first, 452 snapshots
        before the end, but the healthy bearings also sustained alarms
        later; in Test 3 the protocol missed the failed bearing (3,
        outer race) entirely (no sustained alarm), while two healthy
        bearings did alarm. Bearings share one shaft and load, so
        channels are not independent systems.
      </p>
    </Card>
  );
}

export function ResearchView() {
  return (
    <div className="stagger-cards space-y-6">
      <TransfersCard />
      <XaiCard />
      <ImsCard />
    </div>
  );
}
