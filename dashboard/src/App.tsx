import { useState } from "react";
import { data } from "./lib/data";
import { platform } from "./lib/platformData";
import {
  ApiStatusPill,
  useApiHealth,
} from "./components/LiveApiPanel";
import { FleetView } from "./views/FleetView";
import { MachineView } from "./views/MachineView";
import { MaintenanceView } from "./views/MaintenanceView";
import { ModelInfoView } from "./views/ModelInfoView";
import { PlatformView } from "./views/PlatformView";
import { DatasetsView } from "./views/DatasetsView";
import { ExperimentsView } from "./views/ExperimentsView";
import { ResearchView } from "./views/ResearchView";

type View =
  | "fleet"
  | "machine"
  | "maintenance"
  | "model"
  | "platform"
  | "datasets"
  | "experiments"
  | "research";

const NAV: { id: View; label: string; hint: string }[] = [
  { id: "fleet", label: "Fleet Overview", hint: "3 replayed engines" },
  { id: "machine", label: "Machine Detail", hint: "RUL across the replay" },
  { id: "maintenance", label: "Maintenance", hint: "Synthetic demo data" },
  { id: "model", label: "Model Info & Live API", hint: "Metrics, parity, XAI" },
];

/* Phase 8 (Wave 7): multi-dataset platform views, fed by
   dashboard/platform_export.py from platform/experiments/ + the dataset
   registry. Kept as a separate nav group so the original FD001 demo
   navigation above is untouched. */
const NAV_PLATFORM: { id: View; label: string; hint: string }[] = [
  {
    id: "platform",
    label: "Platform",
    hint: "Multi-dataset overview & best results",
  },
  {
    id: "datasets",
    label: "Datasets",
    hint: "Registry — integrated, blocked, researched",
  },
  {
    id: "experiments",
    label: "Experiments",
    hint: "All experiment records, filterable",
  },
  {
    id: "research",
    label: "Research",
    hint: "Transfer, explainability, IMS alarms",
  },
];

const TITLES: Record<View, string> = {
  fleet: "Fleet Overview",
  machine: "Machine Detail",
  maintenance: "Maintenance — Synthetic Demo",
  model: "Model Info & Live API",
  platform: "Platform — Multi-Dataset Overview",
  datasets: "Datasets — Registry",
  experiments: "Experiments — Registry",
  research: "Research — Transfer, XAI, IMS",
};

export default function App() {
  const [view, setView] = useState<View>("fleet");
  const [engine, setEngine] = useState<number>(49);
  const api = useApiHealth();

  function openMachine(eng: number) {
    setEngine(eng);
    setView("machine");
  }

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="flex w-60 shrink-0 flex-col bg-slate-900 text-slate-200">
        <div className="border-b border-slate-800 px-5 py-5">
          <div className="text-xl font-bold tracking-wide text-white">
            PREDICAI
          </div>
          <div className="mt-0.5 text-xs text-slate-400">
            Predict Today. Prevent Tomorrow.
          </div>
          <div className="mt-3 text-[11px] leading-relaxed text-slate-500">
            Predictive maintenance · Remaining Useful Life on NASA C-MAPSS
            FD001 · all RUL values in <strong>cycles</strong>
          </div>
        </div>
        <nav className="flex-1 space-y-1 px-3 py-4">
          {NAV.map((item) => (
            <button
              key={item.id}
              onClick={() => setView(item.id)}
              className={
                view === item.id
                  ? "block w-full rounded border-l-2 border-cyan-400 bg-slate-800 px-3 py-2 text-left"
                  : "block w-full rounded border-l-2 border-transparent px-3 py-2 text-left hover:bg-slate-800/60"
              }
            >
              <span className="block text-sm font-semibold text-slate-100">
                {item.label}
              </span>
              <span className="block text-xs text-slate-500">{item.hint}</span>
            </button>
          ))}
          <div className="px-3 pt-4 pb-1 text-[10px] font-bold tracking-widest text-slate-500 uppercase">
            Platform — Phase 8
          </div>
          {NAV_PLATFORM.map((item) => (
            <button
              key={item.id}
              onClick={() => setView(item.id)}
              className={
                view === item.id
                  ? "block w-full rounded border-l-2 border-cyan-400 bg-slate-800 px-3 py-2 text-left"
                  : "block w-full rounded border-l-2 border-transparent px-3 py-2 text-left hover:bg-slate-800/60"
              }
            >
              <span className="block text-sm font-semibold text-slate-100">
                {item.label}
              </span>
              <span className="block text-xs text-slate-500">{item.hint}</span>
            </button>
          ))}
        </nav>
        <div className="border-t border-slate-800 px-5 py-4 text-[11px] leading-relaxed text-slate-500">
          Data: FD001 replay export + Part 8 synthetic maintenance demo.
          Every panel labels its source.
        </div>
      </aside>

      {/* Main column */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white px-6 py-3">
          <h1 className="text-lg font-bold text-slate-900">
            {TITLES[view]}
          </h1>
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500">
              Rust edge API · localhost:18080
            </span>
            <ApiStatusPill api={api} />
          </div>
        </header>

        <main className="flex-1 px-6 py-6">
          {view === "fleet" && <FleetView onOpen={openMachine} />}
          {view === "machine" && (
            <MachineView engine={engine} onSelect={setEngine} />
          )}
          {view === "maintenance" && <MaintenanceView />}
          {view === "model" && <ModelInfoView api={api} />}
          {view === "platform" && <PlatformView />}
          {view === "datasets" && <DatasetsView />}
          {view === "experiments" && <ExperimentsView />}
          {view === "research" && <ResearchView />}
        </main>

        <footer className="border-t border-slate-200 bg-white px-6 py-3 text-xs text-slate-500">
          Demo build — FD001 replay data; not connected to real machines. ·
          Data exported {data.generated_at} by dashboard/data_export.py from
          simulator/demo.db and ml/artifacts/. · Platform data exported{" "}
          {platform.generated_at} by dashboard/platform_export.py from
          platform/experiments/ and the dataset registry.
        </footer>
      </div>
    </div>
  );
}
