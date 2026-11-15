import { Suspense, lazy } from "react";
import { data, fmt } from "../lib/data";
import { platform } from "../lib/platformData";
import { Card } from "./ui";

const HeroScene = lazy(() => import("./three/HeroScene"));

const CAPTION =
  "TEST CELL · FD001 TURBOFAN (SIMULATED BENCHMARK) · STYLISED VISUALISATION";

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <span className="mono inline-flex items-baseline gap-1.5 rounded border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs dark:border-slate-700 dark:bg-slate-800">
      <span className="font-mono text-[10px] font-medium tracking-[0.14em] text-slate-500 uppercase dark:text-slate-400">
        {label}
      </span>
      <span className="font-semibold text-slate-900 dark:text-slate-100">
        {value}
      </span>
    </span>
  );
}

function ViewportFallback({ label }: { label: string }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-[#05090f] px-4">
      <span className="mono text-center text-[11px] tracking-[0.18em] text-cyan-100/60 uppercase">
        {label}
      </span>
    </div>
  );
}

/**
 * Homepage hero: product headline and measured stats beside a live 3D
 * test-cell viewport. The 3D canvas is lazy-loaded and unmounts while
 * the diagnostic panel is open, so only one WebGL context exists at a
 * time. The viewport itself stays dark in both app themes.
 */
export function HeroPanel({
  onTestMachine,
  sceneActive,
  reducedMotion,
}: {
  onTestMachine: () => void;
  sceneActive: boolean;
  reducedMotion: boolean;
}) {
  const m = data.model.metrics.test_last_window_per_engine;

  const scrollToFleet = () => {
    document
      .getElementById("fleet-engines")
      ?.scrollIntoView({
        behavior: reducedMotion ? "auto" : "smooth",
        block: "start",
      });
  };

  return (
    <Card className="p-5 sm:p-6" ticks>
      <div className="grid items-stretch gap-6 lg:grid-cols-2">
        <div className="flex flex-col">
          <div className="flex items-center gap-2 font-mono text-[11px] font-semibold tracking-[0.18em] text-slate-500 uppercase dark:text-slate-400">
            <span className="inline-block h-2 w-2 shrink-0 bg-cyan-600 dark:bg-cyan-400" />
            Predictive maintenance · Remaining useful life
          </div>
          <h2 className="mt-3 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl dark:text-slate-50">
            Predict Today. Prevent Tomorrow.
          </h2>
          <p className="mt-3 max-w-xl text-sm leading-relaxed text-slate-600 sm:text-[15px] dark:text-slate-300">
            PREDICAI estimates how many operating cycles a turbofan engine
            has left before it needs attention. An LSTM model reads the
            engine's sensor stream and predicts remaining useful life in
            cycles, served by a Rust edge API and tracked across a
            multi-dataset experiment platform.
          </p>

          <div className="mt-4 flex flex-wrap gap-2">
            <StatChip label="Test RMSE" value={`${fmt(m.rmse)} cycles`} />
            <StatChip label="Test MAE" value={`${fmt(m.mae)} cycles`} />
            <StatChip
              label="Engines under replay"
              value={String(data.engines.length)}
            />
            <StatChip
              label="Experiment records"
              value={String(platform.counts.records)}
            />
            <StatChip
              label="Datasets integrated"
              value={String(platform.counts.datasets_integrated)}
            />
          </div>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <button
              onClick={onTestMachine}
              className="inline-flex items-center gap-2 rounded bg-cyan-700 px-4 py-2 text-sm font-semibold text-white transition-colors duration-150 hover:bg-cyan-600"
            >
              <svg
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-4 w-4"
                aria-hidden="true"
              >
                <circle cx="11" cy="11" r="7" />
                <path d="m21 21-4.3-4.3" />
              </svg>
              Test Machine
            </button>
            <button
              onClick={scrollToFleet}
              className="rounded border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 transition-colors duration-150 hover:border-cyan-700 hover:text-cyan-800 dark:border-slate-600 dark:text-slate-200 dark:hover:border-cyan-500 dark:hover:text-cyan-300"
            >
              View engine fleet
            </button>
          </div>
          <p className="mt-3 text-xs text-slate-400 dark:text-slate-500">
            The diagnostic replays recorded FD001 test data. It is not a
            live machine connection.
          </p>
        </div>

        <div className="relative min-h-[280px] overflow-hidden rounded border border-slate-700 bg-[#05090f] lg:min-h-full">
          {sceneActive ? (
            <div className="absolute inset-0">
              <Suspense
                fallback={<ViewportFallback label="Initialising test cell viewport" />}
              >
                <HeroScene
                  mode="idle"
                  progress={0}
                  activeSensor={-1}
                  reducedMotion={reducedMotion}
                />
              </Suspense>
            </div>
          ) : (
            <ViewportFallback label="Test cell view active in the diagnostic panel below" />
          )}
          <div className="mono absolute inset-x-0 bottom-0 bg-[#05090f]/85 px-3 py-1.5 text-[10px] tracking-[0.14em] text-cyan-100/70">
            {CAPTION}
          </div>
        </div>
      </div>
    </Card>
  );
}
