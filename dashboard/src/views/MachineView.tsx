import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { data, fmt, policyForRul } from "../lib/data";
import { Card, LevelChip, SectionTitle, SourceTag, Stat } from "../components/ui";

export function MachineView({
  engine,
  onSelect,
}: {
  engine: number;
  onSelect: (engine: number) => void;
}) {
  const eng = data.engines.find((e) => e.engine === engine) ?? data.engines[0];
  const policy = policyForRul(eng.final.predicted_rul);
  const chartData = eng.series.map((p) => ({
    cycle: p.end_cycle ?? p.step,
    predicted: Number(p.predicted_rul.toFixed(3)),
    true: p.true_rul != null ? Number(p.true_rul.toFixed(1)) : null,
  }));

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-2">
        {data.engines.map((e) => (
          <button
            key={e.engine}
            onClick={() => onSelect(e.engine)}
            className={
              e.engine === eng.engine
                ? "rounded bg-slate-900 px-3 py-1.5 text-sm font-semibold text-white"
                : "rounded border border-slate-300 bg-white px-3 py-1.5 text-sm font-semibold text-slate-600 hover:border-slate-500"
            }
          >
            {e.machine_id}
          </button>
        ))}
        <span className="ml-2">
          <SourceTag label="FD001 REPLAY" />
        </span>
      </div>

      <Card className="p-5">
        <SectionTitle
          title={`${eng.machine_id} — predicted RUL across the replay`}
          source={eng.source}
        />
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="cycle"
                type="number"
                domain={["dataMin", "dataMax"]}
                tickCount={10}
                label={{
                  value: "Recorded cycle (window end)",
                  position: "insideBottom",
                  offset: -2,
                  fontSize: 12,
                  fill: "#64748b",
                }}
                tick={{ fontSize: 12, fill: "#64748b" }}
              />
              <YAxis
                label={{
                  value: "RUL (cycles)",
                  angle: -90,
                  position: "insideLeft",
                  fontSize: 12,
                  fill: "#64748b",
                }}
                tick={{ fontSize: 12, fill: "#64748b" }}
              />
              <Tooltip
                formatter={(value, name) => [
                  `${fmt(Number(value))} cycles`,
                  name === "predicted" ? "Predicted RUL" : "True RUL",
                ]}
                labelFormatter={(label) => `Cycle ${label}`}
              />
              <ReferenceLine
                y={data.policy_thresholds.inspection_max_rul}
                stroke="#d97706"
                strokeDasharray="4 4"
                label={{
                  value: "Inspection ≤ 50",
                  fontSize: 11,
                  fill: "#b45309",
                  // Was "insideTopRight": the label ran past the chart's
                  // right edge and rendered clipped. Anchoring it inside
                  // the top-left keeps the full text within the plot area.
                  position: "insideTopLeft",
                }}
              />
              <ReferenceLine
                y={data.policy_thresholds.review_max_rul}
                stroke="#e11d48"
                strokeDasharray="4 4"
                label={{
                  value: "Review ≤ 30",
                  fontSize: 11,
                  fill: "#be123c",
                  position: "insideBottomRight",
                }}
              />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#0891b2"
                strokeWidth={2}
                dot={chartData.length <= 5}
                activeDot={{ r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="true"
                stroke="#d97706"
                strokeWidth={1.5}
                strokeDasharray="6 4"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-2 flex gap-4 text-xs text-slate-500">
          <span>
            <span className="mr-1 inline-block h-0.5 w-4 bg-cyan-600 align-middle" />
            Predicted RUL (Rust edge API, logged during replay)
          </span>
          <span>
            <span className="mr-1 inline-block h-0.5 w-4 bg-amber-600 align-middle" />
            True RUL (RUL_FD001 definition)
          </span>
        </div>
        {eng.predictions_logged <= 5 && (
          <p className="mt-2 text-xs text-amber-700">
            This test engine has only {eng.total_recorded_cycles} recorded
            cycles, so the replay produced just {eng.predictions_logged}{" "}
            predictions (one per full 30-cycle window). The chart shows
            exactly those points — nothing is interpolated.
          </p>
        )}
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="p-5">
          <SectionTitle title="Replay statistics" />
          <div className="grid grid-cols-2 gap-4">
            <Stat
              label="Predictions logged"
              value={String(eng.predictions_logged)}
            />
            <Stat
              label="Recorded cycles"
              value={String(eng.total_recorded_cycles)}
            />
            <Stat
              label="Predicted RUL range"
              value={`${fmt(eng.stats.min_predicted_rul, 1)} – ${fmt(eng.stats.max_predicted_rul, 1)}`}
              sub="cycles, min – max across the replay"
            />
            <Stat
              label="Mean predicted RUL"
              value={`${fmt(eng.stats.mean_predicted_rul)} cycles`}
            />
            <Stat
              label="Replay MAE vs true"
              value={
                eng.stats.mean_abs_error_vs_true != null
                  ? `${fmt(eng.stats.mean_abs_error_vs_true)} cycles`
                  : "—"
              }
              sub="mean |predicted − true| over all replayed windows"
            />
            <Stat
              label="Cycle alignment"
              value={eng.cycle_alignment_verified ? "Verified" : "Unverified"}
              sub="logged count = recorded cycles − 29, checked at export"
            />
          </div>
        </Card>

        <Card className="p-5">
          <SectionTitle title="Final window summary" />
          <div className="grid grid-cols-2 gap-4">
            <Stat
              label="Final predicted RUL"
              value={`${fmt(eng.final.predicted_rul)} cycles`}
            />
            <Stat
              label="Final true RUL"
              value={`${fmt(eng.final.true_rul, 0)} cycles`}
            />
            <Stat
              label="Final abs. error"
              value={`${fmt(eng.final.abs_error)} cycles`}
            />
            <div>
              <div className="text-xs font-medium tracking-wide text-slate-500 uppercase">
                Policy status (RUL band)
              </div>
              <div className="mt-1.5 flex items-center gap-2">
                <LevelChip level={policy.level} />
                <span className="text-sm text-slate-600">{policy.action}</span>
              </div>
            </div>
          </div>
          <p className="mt-4 border-t border-slate-100 pt-3 text-xs text-slate-500">
            No per-cycle sensor panel is shown for this machine: the API's
            predictions log stores predictions only, not the sensor windows
            behind them. The raw 30 × 14 windows live in the FD001 test file
            (and one is bundled for the live prediction in Model Info) —
            sensor series are not reconstructed or invented here.
          </p>
        </Card>
      </div>
    </div>
  );
}
