import rawData from "../data/demo_data.json";
import type { DemoData } from "./types";

export const data = rawData as unknown as DemoData;

/* Live API access: during `npm run dev` the dashboard calls the same-origin
   /api/* path and Vite proxies it to the Rust API on localhost:18080 (the
   API sends no CORS headers, so direct cross-origin browser calls are
   blocked). Set VITE_API_BASE to override the base URL. */
export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api";
export const API_DISPLAY_URL = "http://localhost:18080";

export type PolicyLevel = "MONITOR" | "INSPECTION" | "REVIEW";

export interface PolicyResult {
  level: PolicyLevel;
  action: string;
}

/** RUL band of maintenance/policy.py (v1.0), applied to a predicted RUL.
    NOTE: these are policy thresholds for planning reviews — NOT validated
    failure boundaries. The full policy also escalates on cycles since
    service; that input only exists for the synthetic demo machine, so the
    fleet/machine views show the RUL band alone and say so. */
export function policyForRul(predictedRul: number): PolicyResult {
  const { review_max_rul, inspection_max_rul } = data.policy_thresholds;
  if (predictedRul <= review_max_rul)
    return { level: "REVIEW", action: "Maintenance review recommended" };
  if (predictedRul <= inspection_max_rul)
    return { level: "INSPECTION", action: "Schedule an inspection" };
  return { level: "MONITOR", action: "Continue monitoring" };
}

export function fmt(n: number, digits = 2): string {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}
