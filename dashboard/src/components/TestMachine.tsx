import {
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { data, fmt, policyForRul } from "../lib/data";
import type { EngineData } from "../lib/types";
import { Card, CountUp, LevelChip, SectionTitle, Stat } from "./ui";

const HeroScene = lazy(() => import("./three/HeroScene"));

type Status = "running" | "assessing" | "done" | "cancelled";
type StageState = "done" | "active" | "pending" | "stopped";

interface LogLine {
  key: number;
  endCycle: number;
  predicted: number;
}

const TICK_MS = 30;
/** Target number of ticks for the longest record; the step adapts so a
    303-cycle record takes roughly 160 ticks (about 5 s). Short records
    finish sooner: the walk is the data, not a fixed-duration timer. */
const TARGET_TICKS = 160;
/** Brief beat between the completed walk and the verdict, so the
    checklist can be read. The progress bar already reads 100% from the
    data at this point; nothing is being computed or fetched. */
const ASSESS_MS = 500;

function StageRow({
  state,
  title,
  detail,
}: {
  state: StageState;
  title: string;
  detail: string;
}) {
  return (
    <li className="flex items-start gap-3">
      <span
        className={`mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-sm border ${
          state === "done"
            ? "border-cyan-600 bg-cyan-600 dark:border-cyan-500 dark:bg-cyan-500"
            : state === "active"
              ? "border-cyan-600 dark:border-cyan-400"
              : "border-slate-300 dark:border-slate-600"
        }`}
      >
        {state === "done" && (
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="white"
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="h-3 w-3"
            aria-hidden="true"
          >
            <path d="M20 6 9 17l-5-5" />
          </svg>
        )}
        {state === "active" && (
          <span className="pulse-dot block h-1.5 w-1.5 rounded-full bg-cyan-600 dark:bg-cyan-400" />
        )}
      </span>
      <span className="min-w-0">
        <span className="block text-sm font-semibold text-slate-800 dark:text-slate-100">
          {title}
        </span>
        <span className="mono block text-xs break-words text-slate-500 dark:text-slate-400">
          {detail}
        </span>
      </span>
    </li>
  );
}

/**
 * Machine diagnostic: replays one engine's recorded FD001 cycles. The
 * scan animation, the progress bar and the stage checklist are all
 * driven by that walk over the recorded data: the cursor advances cycle
 * by cycle, and each logged prediction appears when the cursor passes
 * the cycle it was recorded at. The verdict is the engine's recorded
 * final prediction, banded by the same policy thresholds as the fleet
 * cards. Nothing is fetched and nothing is simulated live.
 */
export function TestMachine({
  engine,
  onSelectEngine,
  onClose,
  onOpen,
  reducedMotion,
}: {
  engine: EngineData;
  onSelectEngine: (engine: number) => void;
  onClose: () => void;
  onOpen: (engine: number) => void;
  reducedMotion: boolean;
}) {
  const total = engine.total_recorded_cycles;
  const series = engine.series;
  const totalPreds = series.length;
  const features = data.model.features;

  const [status, setStatus] = useState<Status>("running");
  const [cursor, setCursor] = useState(0);
  const [emittedCount, setEmittedCount] = useState(0);
  const [log, setLog] = useState<LogLine[]>([]);

  const intervalRef = useRef<number | null>(null);
  const timeoutRef = useRef<number | null>(null);
  const cursorRef = useRef(0);
  const nextIdxRef = useRef(0);

  const clearTimers = useCallback(() => {
    if (intervalRef.current != null) window.clearInterval(intervalRef.current);
    if (timeoutRef.current != null) window.clearTimeout(timeoutRef.current);
    intervalRef.current = null;
    timeoutRef.current = null;
  }, []);

  const start = useCallback(() => {
    clearTimers();
    cursorRef.current = 0;
    nextIdxRef.current = 0;
    setCursor(0);
    setEmittedCount(0);
    setLog([]);

    if (reducedMotion) {
      // No animation loop: the walk resolves immediately and the full
      // recorded result is shown with every stage complete.
      cursorRef.current = total;
      nextIdxRef.current = totalPreds;
      setCursor(total);
      setEmittedCount(totalPreds);
      setLog(
        series.slice(-4).map((p) => ({
          key: p.step,
          endCycle: p.end_cycle ?? total,
          predicted: p.predicted_rul,
        })),
      );
      setStatus("done");
      return;
    }

    setStatus("running");
    const step = Math.max(1, Math.ceil(total / TARGET_TICKS));
    intervalRef.current = window.setInterval(() => {
      const c = Math.min(total, cursorRef.current + step);
      cursorRef.current = c;

      const fresh: LogLine[] = [];
      while (nextIdxRef.current < series.length) {
        const p = series[nextIdxRef.current];
        const due =
          p.end_cycle == null ? c >= total : p.end_cycle <= c;
        if (!due) break;
        fresh.push({
          key: p.step,
          endCycle: p.end_cycle ?? total,
          predicted: p.predicted_rul,
        });
        nextIdxRef.current += 1;
      }

      setCursor(c);
      setEmittedCount(nextIdxRef.current);
      if (fresh.length > 0) {
        setLog((prev) => [...prev, ...fresh].slice(-4));
      }

      if (c >= total) {
        if (intervalRef.current != null)
          window.clearInterval(intervalRef.current);
        intervalRef.current = null;
        setStatus("assessing");
        timeoutRef.current = window.setTimeout(
          () => setStatus("done"),
          ASSESS_MS,
        );
      }
    }, TICK_MS);
  }, [clearTimers, engine, reducedMotion, series, total, totalPreds]);

  // Auto-start on mount. FleetView keys this component by engine, so
  // switching engine tabs remounts it and the walk restarts cleanly.
  useEffect(() => {
    start();
    return clearTimers;
  }, [start, clearTimers]);

  const cancel = useCallback(() => {
    clearTimers();
    setStatus("cancelled");
  }, [clearTimers]);

  const progress = total > 0 ? cursor / total : 0;
  const walking = status === "running" || status === "assessing";
  const activeSensor = walking ? cursor % features.length : -1;
  const sensorName =
    activeSensor >= 0 ? features[activeSensor] : "walk complete";
  const policy = policyForRul(engine.final.predicted_rul);

  const stage1: StageState =
    status === "done" || status === "assessing"
      ? "done"
      : status === "running"
        ? "active"
        : "stopped";
  const stage2 = stage1;
  const stage3: StageState =
    status === "done"
      ? "done"
      : status === "assessing"
        ? "active"
        : status === "cancelled"
          ? "stopped"
          : "pending";

  return (
    <Card className="p-5 sm:p-6" ticks>
      <div className="flex items-start justify-between gap-3">
        <SectionTitle
          title="Machine diagnostic"
          source="Replays the recorded FD001 test data for the selected engine. Animation state follows the recorded cycles and the model's logged predictions."
        />
        <button
          onClick={onClose}
          aria-label="Close diagnostic"
          className="rounded border border-slate-300 p-1.5 text-slate-500 transition-colors duration-150 hover:border-slate-500 hover:text-slate-700 dark:border-slate-600 dark:text-slate-400 dark:hover:border-slate-400 dark:hover:text-slate-200"
        >
          <svg
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            className="h-4 w-4"
            aria-hidden="true"
          >
            <path d="M18 6 6 18" />
            <path d="m6 6 12 12" />
          </svg>
        </button>
      </div>

      <div className="mb-5 flex flex-wrap gap-2">
        {data.engines.map((e) => (
          <button
            key={e.machine_id}
            onClick={() => onSelectEngine(e.engine)}
            className={
              e.engine === engine.engine
                ? "rounded border border-cyan-600 bg-cyan-50 px-3 py-1.5 text-sm font-semibold text-cyan-800 dark:border-cyan-500 dark:bg-cyan-950 dark:text-cyan-300"
                : "rounded border border-slate-300 px-3 py-1.5 text-sm font-semibold text-slate-600 transition-colors duration-150 hover:border-cyan-700 hover:text-cyan-800 dark:border-slate-600 dark:text-slate-300 dark:hover:border-cyan-500 dark:hover:text-cyan-300"
            }
          >
            {e.machine_id}
          </button>
        ))}
      </div>

      <div className="grid items-stretch gap-6 lg:grid-cols-2">
        <div className="relative min-h-[224px] overflow-hidden rounded border border-slate-700 bg-[#05090f]">
          <div className="absolute inset-0">
            <Suspense
              fallback={
                <div className="flex h-full items-center justify-center px-4">
                  <span className="mono text-center text-[11px] tracking-[0.18em] text-cyan-100/60 uppercase">
                    Initialising test cell viewport
                  </span>
                </div>
              }
            >
              <HeroScene
                mode={walking ? "scanning" : "idle"}
                progress={progress}
                activeSensor={activeSensor}
                reducedMotion={reducedMotion}
              />
            </Suspense>
          </div>
          <div className="mono absolute top-2 left-2 rounded bg-[#05090f]/85 px-2 py-1 text-[11px] text-cyan-100/90">
            CYCLE {cursor} / {total}
            {walking && <span> · {sensorName}</span>}
          </div>
          <div className="mono absolute inset-x-0 bottom-0 bg-[#05090f]/85 px-3 py-1.5 text-[10px] tracking-[0.14em] text-cyan-100/70">
            TEST CELL · DIAGNOSTIC SCAN · RECORDED FD001 DATA
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div>
            <div className="mb-1.5 flex items-baseline justify-between gap-3">
              <span className="font-mono text-[11px] font-medium tracking-[0.14em] text-slate-500 uppercase dark:text-slate-400">
                Replay progress
              </span>
              <span className="mono text-sm font-semibold text-slate-800 dark:text-slate-100">
                {(progress * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded bg-slate-200 dark:bg-slate-800">
              <div
                className="h-full bg-cyan-600 dark:bg-cyan-400"
                style={{ width: `${progress * 100}%` }}
              />
            </div>
          </div>

          <ol className="space-y-3">
            <StageRow
              state={stage1}
              title="Sensor stream"
              detail={
                status === "cancelled"
                  ? `Stopped at cycle ${cursor} of ${total}`
                  : stage1 === "done"
                    ? `${total} recorded cycles replayed`
                    : `Cycle ${cursor} of ${total} · scanning ${sensorName}`
              }
            />
            <StageRow
              state={stage2}
              title="Model inference"
              detail={`${emittedCount} of ${totalPreds} logged predictions processed`}
            />
            <StageRow
              state={stage3}
              title="Health assessment"
              detail={
                status === "done"
                  ? `Verdict: ${policy.level} · ${policy.action}`
                  : status === "assessing"
                    ? "Applying the policy RUL bands to the final prediction"
                    : status === "cancelled"
                      ? "Not reached"
                      : "Starts when the walk completes"
              }
            />
          </ol>

          <div className="rounded border border-slate-200 bg-slate-50 px-3 py-2.5 dark:border-slate-700 dark:bg-slate-800">
            <div className="font-mono text-[11px] font-medium tracking-[0.14em] text-slate-500 uppercase dark:text-slate-400">
              Prediction log (latest)
            </div>
            {log.length === 0 ? (
              <p className="mono mt-1.5 text-xs text-slate-500 dark:text-slate-400">
                No predictions yet. The first logged prediction lands at
                cycle 30, once the first full input window is available.
              </p>
            ) : (
              <ul className="mono mt-1.5 space-y-1 text-xs text-slate-700 dark:text-slate-200">
                {log.map((l) => (
                  <li key={l.key}>
                    cycle {l.endCycle} · predicted RUL {fmt(l.predicted)}{" "}
                    cycles
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      {status === "done" && (
        <div className="mt-6 border-t border-slate-100 pt-5 dark:border-slate-800">
          <div className="grid items-start gap-5 md:grid-cols-[1fr_auto]">
            <div>
              <div className="tabular rounded border border-slate-200 bg-slate-50 px-3 py-2.5 dark:border-slate-700 dark:bg-slate-800">
                <div className="font-mono text-[11px] font-medium tracking-[0.14em] text-slate-500 uppercase dark:text-slate-400">
                  Final predicted RUL
                </div>
                <div className="mono mt-0.5 text-3xl font-bold text-cyan-800 sm:text-4xl dark:text-cyan-300">
                  <CountUp value={engine.final.predicted_rul} format={fmt} />{" "}
                  <span className="text-base font-semibold text-slate-500 dark:text-slate-400">
                    cycles
                  </span>
                </div>
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-2">
                <LevelChip level={policy.level} />
                <span className="text-sm text-slate-600 dark:text-slate-400">
                  {policy.action}
                </span>
              </div>
              <p className="mt-3 max-w-2xl text-xs leading-relaxed text-slate-400 dark:text-slate-500">
                Diagnostic replayed the recorded FD001 test data for this
                engine; predictions are the model's logged outputs, not a
                live machine connection.
              </p>
            </div>
            <div className="grid grid-cols-2 gap-x-8 gap-y-4 md:grid-cols-1">
              <Stat
                label="True RUL (final)"
                value={`${fmt(engine.final.true_rul, 0)} cycles`}
              />
              <Stat
                label="Final abs. error"
                value={`${fmt(engine.final.abs_error)} cycles`}
              />
            </div>
          </div>
          <div className="mt-5 flex flex-wrap gap-3">
            <button
              onClick={() => onOpen(engine.engine)}
              className="rounded bg-cyan-700 px-4 py-2 text-sm font-semibold text-white transition-colors duration-150 hover:bg-cyan-600"
            >
              Open full results
            </button>
            <button
              onClick={start}
              className="rounded border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 transition-colors duration-150 hover:border-cyan-700 hover:text-cyan-800 dark:border-slate-600 dark:text-slate-200 dark:hover:border-cyan-500 dark:hover:text-cyan-300"
            >
              Run again
            </button>
          </div>
        </div>
      )}

      {status === "cancelled" && (
        <div className="mt-6 border-t border-slate-100 pt-5 dark:border-slate-800">
          <p className="mono text-sm text-slate-600 dark:text-slate-300">
            Diagnostic cancelled at cycle {cursor} of {total} ·{" "}
            {emittedCount} of {totalPreds} predictions processed.
          </p>
          <div className="mt-4 flex flex-wrap gap-3">
            <button
              onClick={start}
              className="rounded bg-cyan-700 px-4 py-2 text-sm font-semibold text-white transition-colors duration-150 hover:bg-cyan-600"
            >
              Run diagnostic
            </button>
          </div>
        </div>
      )}

      {walking && (
        <div className="mt-6 flex flex-wrap gap-3 border-t border-slate-100 pt-5 dark:border-slate-800">
          <button
            onClick={cancel}
            className="rounded border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 transition-colors duration-150 hover:border-rose-600 hover:text-rose-700 dark:border-slate-600 dark:text-slate-200 dark:hover:border-rose-500 dark:hover:text-rose-300"
          >
            Cancel diagnostic
          </button>
        </div>
      )}
    </Card>
  );
}
