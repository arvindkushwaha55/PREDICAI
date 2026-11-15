import { data, fmt } from "../lib/data";
import { Card, LevelChip, SectionTitle } from "../components/ui";

export function MaintenanceView() {
  const m = data.maintenance;
  return (
    <div className="stagger-cards space-y-6">
      <div className="rounded-lg border-2 border-amber-500 bg-amber-50 dark:bg-amber-950 px-5 py-4">
        <div className="text-sm font-bold tracking-wide text-amber-900 dark:text-amber-200">
          SYNTHETIC DEMO DATA
        </div>
        <p className="mt-1 text-sm text-amber-900 dark:text-amber-200">
          NASA C-MAPSS FD001 contains <strong>no real maintenance
          records</strong>. Everything on this page is the Part 8 maintenance
          module's clearly-flagged synthetic demo for {m.machine_id}; it
          demonstrates the history + recommendation workflow, and must never
          be read as a real service history.
        </p>
      </div>

      <Card className="p-5" ticks>
        <SectionTitle
          title={`Service history: ${m.machine_id}`}
          source={m.source}
        />
        {m.machine && (
          <p className="mb-3 text-sm text-slate-600 dark:text-slate-400">
            {m.machine.display_name} · {m.machine.notes}
          </p>
        )}
        {m.service_records.length === 0 ? (
          <p className="text-sm text-slate-500 dark:text-slate-400">No service records.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="tabular mono w-full min-w-[620px] text-left text-sm">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-700 text-xs tracking-wide text-slate-500 dark:text-slate-400 uppercase">
                  <th className="py-2 pr-4">Date</th>
                  <th className="py-2 pr-4">At cycle</th>
                  <th className="py-2 pr-4">Type</th>
                  <th className="py-2 pr-4">Notes</th>
                  <th className="py-2">Flag</th>
                </tr>
              </thead>
              <tbody>
                {m.service_records.map((r) => (
                  <tr
                    key={r.record_id}
                    className="border-b border-slate-100 dark:border-slate-800 align-top transition-colors duration-150 hover:bg-slate-50 dark:hover:bg-slate-800"
                  >
                    <td className="tabular py-2 pr-4">{r.service_date}</td>
                    <td className="tabular py-2 pr-4">{r.service_cycle}</td>
                    <td className="py-2 pr-4 font-medium text-slate-800 dark:text-slate-200">
                      {r.service_type}
                    </td>
                    <td className="py-2 pr-4 text-slate-600 dark:text-slate-400">{r.notes}</td>
                    <td className="py-2">
                      {r.is_synthetic === 1 ? (
                        <span className="inline-flex rounded border border-amber-600/40 bg-amber-100 dark:bg-amber-950 px-1.5 py-0.5 text-[11px] font-bold text-amber-900 dark:text-amber-200">
                          SYNTHETIC
                        </span>
                      ) : (
                        <span className="text-slate-500 dark:text-slate-400">real</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
          Dates and cycles belong to the synthetic demo machine's own
          timeline. Rules from the maintenance module: service history is
          never an input to the LSTM, and a service event never resets the
          model's RUL.
        </p>
      </Card>

      <Card className="p-5">
        <SectionTitle
          title="Recommendations"
          source="Produced by maintenance/policy.py (policy v1.0) through the module's make_recommendation(): the reason text below is the module's own output, citing its actual inputs."
        />
        {m.recommendations.length === 0 ? (
          <p className="text-sm text-slate-500 dark:text-slate-400">No recommendations.</p>
        ) : (
          <div className="space-y-4">
            {m.recommendations.map((r) => (
              <div
                key={r.rec_id}
                className="rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 px-4 py-3"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <LevelChip level={r.level} />
                  <span className="tabular text-xs text-slate-500 dark:text-slate-400">
                    policy v{r.policy_version} · {r.created_at} · inputs:
                    predicted RUL{" "}
                    {r.predicted_rul != null
                      ? `${fmt(r.predicted_rul)} cycles`
                      : "-"}
                    , current cycle {r.current_cycle ?? "-"},{" "}
                    {r.cycles_since_service ?? "-"} cycles since service
                  </span>
                </div>
                <p className="mt-2 text-sm text-slate-800 dark:text-slate-200">{r.reason}</p>
              </div>
            ))}
          </div>
        )}
        <div className="tabular mt-4 border-t border-slate-100 dark:border-slate-800 pt-3 text-xs text-slate-500 dark:text-slate-400">
          Inputs used: predicted RUL ={" "}
          {fmt(m.recommendation_inputs.predicted_rul)} cycles (
          {m.recommendation_inputs.predicted_rul_source}); current cycle ={" "}
          {m.recommendation_inputs.current_cycle} (
          {m.recommendation_inputs.current_cycle_source}). Thresholds are
          policy choices, not validated failure boundaries.
        </div>
      </Card>
    </div>
  );
}
