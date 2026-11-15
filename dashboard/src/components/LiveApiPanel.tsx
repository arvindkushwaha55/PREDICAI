import { useCallback, useEffect, useState } from "react";
import { API_BASE, API_DISPLAY_URL, data, fmt } from "../lib/data";
import type { HealthResponse, PredictResponse } from "../lib/types";
import { Card, SectionTitle } from "./ui";

export type ApiStatus = "checking" | "live" | "offline";

export interface ApiHealthState {
  status: ApiStatus;
  health: HealthResponse | null;
  error: string | null;
  refresh: () => void;
}

/** Polls GET /health once on mount (and on demand). No fake fallback: any
    failure is reported as OFFLINE with the reason. */
export function useApiHealth(): ApiHealthState {
  const [status, setStatus] = useState<ApiStatus>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setStatus("checking");
    fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) })
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        setHealth((await res.json()) as HealthResponse);
        setError(null);
        setStatus("live");
      })
      .catch((e: unknown) => {
        setHealth(null);
        setError(e instanceof Error ? e.message : String(e));
        setStatus("offline");
      });
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { status, health, error, refresh };
}

export function ApiStatusPill({ api }: { api: ApiHealthState }) {
  const styles =
    api.status === "live"
      ? "bg-emerald-100 dark:bg-emerald-950 text-emerald-800 dark:text-emerald-300 border-emerald-600/30"
      : api.status === "offline"
        ? "bg-rose-100 dark:bg-rose-950 text-rose-800 dark:text-rose-300 border-rose-600/30"
        : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400 border-slate-400/40";
  const dot =
    api.status === "live"
      ? "bg-emerald-500"
      : api.status === "offline"
        ? "bg-rose-500"
        : "bg-slate-400";
  const label =
    api.status === "live"
      ? "API LIVE"
      : api.status === "offline"
        ? "API OFFLINE"
        : "API CHECKING…";
  return (
    <button
      onClick={api.refresh}
      title={`Rust edge API at ${API_DISPLAY_URL}. Click to re-check`}
      className={`inline-flex items-center gap-2 rounded border px-2.5 py-1 text-xs font-bold tracking-wide transition-colors duration-150 ${styles}`}
    >
      <span className={`h-2 w-2 rounded-full ${dot} ${api.status === "live" ? "pulse-dot" : ""}`} />
      {label}
    </button>
  );
}

export function LiveApiPanel({ api }: { api: ApiHealthState }) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [predictError, setPredictError] = useState<string | null>(null);

  const sample = data.live_sample;
  const canPredict =
    api.status === "live" && api.health?.model_loaded === true && !running;

  async function runPrediction() {
    setRunning(true);
    setPredictError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/predict_rul`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          machine_id: sample.machine_id,
          window: sample.window,
        }),
        signal: AbortSignal.timeout(15000),
      });
      const body: unknown = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(
          `HTTP ${res.status}${body ? `: ${JSON.stringify(body)}` : ""}`,
        );
      }
      setResult(body as PredictResponse);
    } catch (e: unknown) {
      setPredictError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <Card className="p-5" ticks>
      <SectionTitle
        title="Live API: Rust edge engine"
        source={`GET /health and POST /predict_rul against the Rust API at ${API_DISPLAY_URL} (via the Vite dev proxy /api). Status below is measured in your browser, not assumed.`}
      />

      <div className="flex flex-wrap items-center gap-3">
        <ApiStatusPill api={api} />
        {api.status === "live" && api.health && (
          <span className="tabular mono text-sm text-slate-600 dark:text-slate-400">
            model_loaded:{" "}
            <strong>{api.health.model_loaded ? "true" : "false"}</strong> ·
            window {api.health.seq_len} cycles × {api.health.num_features}{" "}
            sensors
            {api.health.model_path && (
              <>
                {" "}
                · <span className="text-slate-500 dark:text-slate-400">{api.health.model_path}</span>
              </>
            )}
          </span>
        )}
        {api.status === "offline" && (
          <span className="text-sm text-rose-700 dark:text-rose-300">
            Unreachable{api.error ? ` (${api.error})` : ""}. Start the edge
            server on port 18080 and view this dashboard via{" "}
            <code className="rounded bg-slate-100 dark:bg-slate-800 px-1">npm run dev</code>:
            the API sends no CORS headers, so a statically served build
            cannot read it from the browser.
          </span>
        )}
        {api.status === "live" && api.health && !api.health.model_loaded && (
          <span className="text-sm text-amber-700 dark:text-amber-300">
            API is up but the ONNX model is not loaded: predictions would
            return 503, so the button stays disabled.
          </span>
        )}
      </div>

      <div className="mt-4 border-t border-slate-100 dark:border-slate-800 pt-4">
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={runPrediction}
            disabled={!canPredict}
            className={
              canPredict
                ? "rounded bg-cyan-700 px-4 py-2 text-sm font-semibold text-white hover:bg-cyan-800"
                : "cursor-not-allowed rounded bg-slate-200 dark:bg-slate-700 px-4 py-2 text-sm font-semibold text-slate-500 dark:text-slate-400"
            }
          >
            {running ? "Running…" : "Run live prediction"}
          </button>
          <span className="text-sm text-slate-600 dark:text-slate-400">
            Sends the bundled ENGINE-001 sample window (30 × 14 raw sensor
            values, real FD001 test data) to the live model. Fixture
            expectation: ≈ {fmt(sample.expected_rul_from_parity_fixture)}{" "}
            cycles (true RUL {fmt(sample.true_rul, 0)}).
          </span>
        </div>
        {api.status !== "live" && (
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
            Live prediction is unavailable while the API is offline; no
            substitute numbers are shown.
          </p>
        )}
        {predictError && (
          <p className="mt-3 text-sm font-medium text-rose-700 dark:text-rose-300">
            Prediction failed: {predictError}
          </p>
        )}
        {result && (
          <div className="tabular mt-3 rounded border border-emerald-600/30 bg-emerald-50 dark:bg-emerald-950 px-4 py-3">
            <span className="text-sm text-emerald-900 dark:text-emerald-200">
              Live result: {result.machine_id}:{" "}
              <strong className="mono text-lg">
                {fmt(result.predicted_rul)} {result.unit}
              </strong>{" "}
              predicted RUL · status {result.status} · Δ vs parity fixture{" "}
              {fmt(
                Math.abs(
                  result.predicted_rul -
                    sample.expected_rul_from_parity_fixture,
                ),
                4,
              )}{" "}
              cycles
            </span>
          </div>
        )}
        <p className="mt-2 text-xs text-slate-400 dark:text-slate-500">{sample.source}</p>
      </div>
    </Card>
  );
}
