import { platform, datasetLabel, fmtMetric } from "../lib/platformData";
import { Card, CountUp, SectionTitle, Stat } from "../components/ui";

const TASK_NAMES: Record<string, string> = {
  A: "Task A: Failure / fault classification",
  B: "Task B: Remaining Useful Life (regression)",
  C: "Task C: Anomaly detection",
};

export function PlatformView() {
  const c = platform.counts;
  const tasksCovered = Object.keys(c.tasks).sort();
  return (
    <div className="stagger-cards space-y-6">
      <Card className="p-5" ticks>
        <SectionTitle
          title="Multi-dataset platform: overview"
          source={platform.source}
        />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat
            label="Datasets integrated"
            value={<CountUp value={c.datasets_integrated} format={(n) => `${Math.round(n)} of ${c.datasets_registry_entries}`} />}
            sub="registry entries incl. blocked / researched-only"
          />
          <Stat
            label="Experiment records"
            value={<CountUp value={c.records} format={(n) => String(Math.round(n))} />}
            sub={`${c.transfers} cross-dataset transfers · ${c.explainability} explainability records`}
          />
          <Stat
            label="Trained-model records"
            value={<CountUp value={c.trained_model_records} format={(n) => String(Math.round(n))} />}
            sub="excl. zero-shot transfers and explainability records"
          />
          <Stat
            label="Tasks covered"
            value={tasksCovered.join(" · ")}
            sub={tasksCovered
              .map((t) => `${t}: ${c.tasks[t]} records`)
              .join(" · ")}
          />
        </div>
        <p className="mt-4 border-t border-slate-100 dark:border-slate-800 pt-3 text-xs text-slate-500 dark:text-slate-400">
          Every number on the platform pages comes from an experiment
          record, exported {platform.generated_at}. Scores
          from different datasets or protocols are never averaged together,
          and a record's caveats travel with its score; see the Research
          view for the cases where the honest headline is the weaker
          number.
        </p>
      </Card>

      {tasksCovered.map((task) => {
        const rows = platform.best_results.filter((b) => b.task === task);
        if (rows.length === 0) return null;
        return (
          <Card key={task} className="p-5">
            <SectionTitle
              title={`Best verified result per dataset: ${TASK_NAMES[task] ?? task}`}
              source="Winner per dataset×task by the metric named on the row, computed at export time from the records. Cross-dataset score comparison is not meaningful; each row stands on its own protocol."
            />
            <div className="overflow-x-auto">
              <table className="tabular mono w-full min-w-[600px] text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-700 text-xs tracking-wide text-slate-500 dark:text-slate-400 uppercase">
                    <th className="py-2 pr-4">Dataset</th>
                    <th className="py-2 pr-4">Model</th>
                    <th className="py-2 pr-4">Metric</th>
                    <th className="py-2 pr-4">Value</th>
                    <th className="py-2">Record</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((b) => (
                    <tr
                      key={b.experiment_id}
                      className="border-b border-slate-100 dark:border-slate-800 transition-colors duration-150 hover:bg-slate-50 dark:hover:bg-slate-800"
                    >
                      <td className="py-2 pr-4 font-medium text-slate-800 dark:text-slate-200">
                        {datasetLabel(b.dataset)}
                      </td>
                      <td className="py-2 pr-4">{b.model}</td>
                      <td className="py-2 pr-4 text-slate-600 dark:text-slate-400">
                        {b.metric_label}
                      </td>
                      <td className="py-2 pr-4 font-semibold">
                        {b.metric_value != null
                          ? fmtMetric(b.metric_value)
                          : (b.metric_display ?? "-")}
                      </td>
                      <td className="py-2 font-mono text-xs text-slate-500 dark:text-slate-400">
                        {b.experiment_id}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {task === "A" && (
              <p className="mt-3 text-xs text-amber-700 dark:text-amber-300">
                CWRU note: the wave-2 window-split record (accuracy 1.000)
                is deliberately NOT the headline: windows from the same
                recording sat in both splits. The recording-wise
                leave-one-recording-out record above (F1-macro 0.689,
                order-domain features, Wave 12) is the honest CWRU
                result.
              </p>
            )}
            {task === "B" && (
              <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
                FD001 note: the GRU row is a local CPU architecture
                comparison on the shipped pipeline's protocol. The deployed
                model remains the owner's Kaggle-trained LSTM (test RMSE
                13.73) shown in Model Info; a single-seed comparison does
                not replace it.
              </p>
            )}
          </Card>
        );
      })}
    </div>
  );
}
