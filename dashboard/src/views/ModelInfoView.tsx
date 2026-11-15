import { data, fmt } from "../lib/data";
import type { MetricSet } from "../lib/types";
import {
  LiveApiPanel,
  type ApiHealthState,
} from "../components/LiveApiPanel";
import { Card, CountUp, SectionTitle, Stat } from "../components/ui";

function MetricsTable() {
  const rows: { label: string; m: MetricSet }[] = [
    {
      label: "Test: last window per engine (official protocol)",
      m: data.model.metrics.test_last_window_per_engine,
    },
    {
      label: "Validation: all windows (engine-wise holdout)",
      m: data.model.metrics.validation_all_windows,
    },
  ];
  return (
    <div className="overflow-x-auto">
      <table className="tabular mono w-full text-left text-sm">
        <thead>
          <tr className="border-b border-slate-200 dark:border-slate-700 text-xs tracking-wide text-slate-500 dark:text-slate-400 uppercase">
            <th className="py-2 pr-4">Split</th>
            <th className="py-2 pr-4">RMSE (cycles)</th>
            <th className="py-2 pr-4">MAE (cycles)</th>
            <th className="py-2 pr-4">NASA PHM08 score</th>
            <th className="py-2">n</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="border-b border-slate-100 dark:border-slate-800 transition-colors duration-150 hover:bg-slate-50 dark:hover:bg-slate-800">
              <td className="py-2 pr-4 font-medium text-slate-800 dark:text-slate-200">
                {r.label}
              </td>
              <td className="py-2 pr-4 font-semibold">{fmt(r.m.rmse)}</td>
              <td className="py-2 pr-4">{fmt(r.m.mae)}</td>
              <td className="py-2 pr-4">{fmt(r.m.nasa_score, 1)}</td>
              <td className="py-2">{r.m.n.toLocaleString("en-US")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ModelInfoView({ api }: { api: ApiHealthState }) {
  const mdl = data.model;
  const xai = mdl.explainability;
  const tc = mdl.training_config;
  const SPEC_ROWS: [string, string][] = [
    ["Architecture", mdl.architecture],
    ["Input window", `${mdl.seq_len} cycles \u00d7 ${mdl.num_features} sensors`],
    ["RUL cap (training)", `${mdl.rul_cap} cycles`],
    ["LSTM hidden units", `${tc.hidden_1} / ${tc.hidden_2}`],
    ["Head dimension", String(tc.head_dim)],
    ["Dropout", String(tc.dropout)],
    ["Learning rate", String(tc.lr)],
    ["Weight decay", String(tc.weight_decay)],
    ["Batch size", String(tc.batch_size)],
    ["Max epochs", String(tc.epochs)],
    ["Early-stop patience", `${tc.patience} epochs`],
    ["Gradient clip", String(tc.grad_clip)],
    ["Validation fraction", String(tc.val_fraction)],
    ["ONNX opset", String(mdl.onnx_opset)],
    ["Seed", String(mdl.seed)],
    ["Weights file", mdl.weights],
    ["Serving file", mdl.model_file],
  ];

  return (
    <div className="stagger-cards space-y-6">
      <LiveApiPanel api={api} />

      <Card className="p-5" ticks>
        <SectionTitle title="Model" source={mdl.source} />
        <div className="grid gap-4 md:grid-cols-3">
          <Stat label="Architecture" value={mdl.architecture} />
          <Stat label="Dataset" value={mdl.dataset} />
          <Stat
            label="Input window"
            value={`${mdl.seq_len} cycles × ${mdl.num_features} sensors`}
          />
          <Stat label="RUL cap (training)" value={`${mdl.rul_cap} cycles`} />
          <Stat label="Seed" value={String(mdl.seed)} />
          <Stat label="Trained on" value={mdl.trained_on} />
          <Stat
            label="Serving format"
            value={`${mdl.model_file} · ONNX opset ${mdl.onnx_opset}`}
          />
          <Stat label="Weights" value={mdl.weights} />
          <Stat
            label="Best val RMSE (training)"
            value={<CountUp value={mdl.metrics.best_val_rmse_during_training} format={(n) => `${fmt(n)} cycles`} />}
          />
        </div>
        <div className="mt-4 border-t border-slate-100 dark:border-slate-800 pt-3">
          <div className="text-xs font-medium tracking-wide text-slate-500 dark:text-slate-400 uppercase">
            Features ({mdl.num_features} sensors, constant sensors dropped in
            Part 3)
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {mdl.features.map((f) => (
              <span
                key={f}
                className="rounded bg-slate-100 dark:bg-slate-800 px-2 py-0.5 font-mono text-xs text-slate-700 dark:text-slate-300"
              >
                {f}
              </span>
            ))}
          </div>
        </div>
      </Card>

      <Card className="p-5" ticks>
        <SectionTitle
          title="Technical specification"
          source="Architecture and training configuration as recorded in ml/artifacts (model_metadata.json and the run's training config) for the shipped model. Values are read from the export, not re-measured in the browser."
        />
        <div className="grid gap-x-10 sm:grid-cols-2">
          {SPEC_ROWS.map(([k, v]) => (
            <div
              key={k}
              className="flex items-baseline justify-between gap-4 border-b border-slate-100 py-1.5 dark:border-slate-800"
            >
              <span className="font-mono text-[11px] font-medium tracking-wider text-slate-500 uppercase dark:text-slate-400">
                {k}
              </span>
              <span className="mono text-right text-[13px] font-medium text-slate-800 dark:text-slate-200">
                {v}
              </span>
            </div>
          ))}
        </div>
      </Card>

      <Card className="p-5">
        <SectionTitle
          title="Measured performance: real run only"
          source="metrics.json from the Kaggle training run; independently re-verified by recomputing from the raw predictions file (exact match). RUL is always in cycles."
        />
        <MetricsTable />
      </Card>

      <Card className="p-5">
        <SectionTitle title="Cross-stack parity (measured)" />
        <div className="tabular space-y-2 text-sm text-slate-700 dark:text-slate-300">
          <p>
            PyTorch ↔ ONNX max abs diff:{" "}
            <strong>
              {mdl.parity.pytorch_vs_onnx_max_abs_diff.toExponential(2)}
            </strong>{" "}
            cycles (tolerance 1e-4).
          </p>
          <p>Rust ↔ Python ONNX: {mdl.parity.rust_vs_python_onnx}.</p>
        </div>
      </Card>

      <Card className="p-5">
        <SectionTitle
          title="Explainability: which sensors the model relies on"
          source={`Part 6 explainability on the ${xai.split} (n=${xai.n_val_windows.toLocaleString("en-US")} windows). Two independent methods, shown side by side.`}
        />
        <div className="grid gap-6 md:grid-cols-2">
          <div>
            <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
              Permutation importance: top 5
            </h3>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              Validation RMSE increase when that sensor's trace is shuffled.
            </p>
            <ol className="tabular mt-2 space-y-1.5">
              {xai.permutation_top5.map((r, i) => (
                <li
                  key={r.sensor}
                  className="flex items-baseline justify-between border-b border-slate-100 dark:border-slate-800 pb-1 text-sm"
                >
                  <span>
                    <span className="mr-2 text-slate-400 dark:text-slate-500">{i + 1}.</span>
                    <span className="font-mono">{r.sensor}</span>
                  </span>
                  <span className="font-semibold">
                    +{fmt(r.rmse_increase)} cycles RMSE
                  </span>
                </li>
              ))}
            </ol>
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-200">
              Integrated Gradients: top 5
            </h3>
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
              Mean |attribution| in cycles of predicted RUL, baseline = train
              mean operating point.
            </p>
            <ol className="tabular mt-2 space-y-1.5">
              {xai.integrated_gradients_top5.map((r, i) => (
                <li
                  key={r.sensor}
                  className="flex items-baseline justify-between border-b border-slate-100 dark:border-slate-800 pb-1 text-sm"
                >
                  <span>
                    <span className="mr-2 text-slate-400 dark:text-slate-500">{i + 1}.</span>
                    <span className="font-mono">{r.sensor}</span>
                  </span>
                  <span className="font-semibold">
                    {fmt(r.mean_abs_attribution, 3)} cycles
                  </span>
                </li>
              ))}
            </ol>
          </div>
        </div>
        <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">
          Sensors in both top-5 lists:{" "}
          <strong>{xai.top5_overlap.join(", ")}</strong>. The rankings
          otherwise differ; expected, since the methods measure different
          things (accuracy reliance vs. per-prediction attribution).
        </p>
        <p className="mt-2 rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 px-3 py-2 text-xs text-slate-600 dark:text-slate-400">
          {xai.note}
        </p>
      </Card>
    </div>
  );
}
