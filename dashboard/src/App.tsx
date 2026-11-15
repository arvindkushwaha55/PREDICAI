import { useEffect, useState } from "react";
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
import { PrivacyView } from "./views/PrivacyView";
import { TermsView } from "./views/TermsView";

type View =
  | "fleet"
  | "machine"
  | "maintenance"
  | "model"
  | "platform"
  | "datasets"
  | "experiments"
  | "research"
  | "privacy"
  | "terms";

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
    hint: "Registry: integrated, blocked, researched",
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

const NAV_LEGAL: { id: View; label: string }[] = [
  { id: "privacy", label: "Privacy Policy" },
  { id: "terms", label: "Terms and Conditions" },
];

const TITLES: Record<View, string> = {
  fleet: "Fleet Overview",
  machine: "Machine Detail",
  maintenance: "Maintenance: Synthetic Demo",
  model: "Model Info & Live API",
  platform: "Platform: Multi-Dataset Overview",
  datasets: "Datasets: Registry",
  experiments: "Experiments: Registry",
  research: "Research: Transfer, XAI, IMS",
  privacy: "Privacy Policy",
  terms: "Terms and Conditions",
};

type Theme = "light" | "dark";

function initialTheme(): Theme {
  try {
    return localStorage.getItem("predicai-theme") === "dark"
      ? "dark"
      : "light";
  } catch {
    return "light";
  }
}

function MoonIcon() {
  return (
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
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z" />
    </svg>
  );
}

function MenuIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-5 w-5"
      aria-hidden="true"
    >
      <path d="M4 6h16" />
      <path d="M4 12h16" />
      <path d="M4 18h16" />
    </svg>
  );
}

function CloseIcon() {
  return (
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
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </svg>
  );
}

function SunIcon() {
  return (
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
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2" />
      <path d="M12 20v2" />
      <path d="m4.93 4.93 1.41 1.41" />
      <path d="m17.66 17.66 1.41 1.41" />
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="m6.34 17.66-1.41 1.41" />
      <path d="m19.07 4.93-1.41 1.41" />
    </svg>
  );
}

export default function App() {
  const [view, setView] = useState<View>("fleet");
  const [engine, setEngine] = useState<number>(49);
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [navOpen, setNavOpen] = useState(false);
  const api = useApiHealth();

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    try {
      localStorage.setItem("predicai-theme", theme);
    } catch {
      // Storage blocked (private mode etc.): the toggle still applies
      // for this session, it just is not remembered.
    }
  }, [theme]);

  /* While the mobile drawer is open, lock body scroll and let Escape
     close it. Cleanup restores the previous overflow value. */
  useEffect(() => {
    if (!navOpen) return;
    const prev = document.documentElement.style.overflow;
    document.documentElement.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setNavOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.documentElement.style.overflow = prev;
      document.removeEventListener("keydown", onKey);
    };
  }, [navOpen]);

  function openMachine(eng: number) {
    setEngine(eng);
    setView("machine");
  }

  const navButton = (item: { id: View; label: string; hint?: string }) => (
    <button
      key={item.id}
      onClick={() => {
        setView(item.id);
        setNavOpen(false);
      }}
      className={
        view === item.id
          ? "block w-full rounded border-l-2 border-cyan-400 bg-slate-800 px-3 py-2 text-left transition-colors duration-150"
          : "block w-full rounded border-l-2 border-transparent px-3 py-2 text-left transition-colors duration-150 hover:bg-slate-800/60"
      }
    >
      <span className="block text-sm font-semibold text-slate-100">
        {item.label}
      </span>
      {item.hint && (
        <span className="block text-xs text-slate-500">{item.hint}</span>
      )}
    </button>
  );

  /* Sidebar inner content (brand block, nav groups, footer note) is a
     single shared block: the desktop aside and the mobile slide-over
     drawer both render it, so the two navigations cannot drift apart. */
  const sidebarContent = (
    <>
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
        {NAV.map(navButton)}
        <div className="px-3 pt-4 pb-1 font-mono text-[10px] font-bold tracking-widest text-slate-500 uppercase">
          Platform · Phase 8
        </div>
        {NAV_PLATFORM.map(navButton)}
        <div className="px-3 pt-4 pb-1 font-mono text-[10px] font-bold tracking-widest text-slate-500 uppercase">
          Legal
        </div>
        {NAV_LEGAL.map(navButton)}
      </nav>
      <div className="border-t border-slate-800 px-5 py-4 text-[11px] leading-relaxed text-slate-500">
        Data: FD001 replay export + Part 8 synthetic maintenance demo.
        Every panel labels its source.
      </div>
    </>
  );

  return (
    <div className="flex min-h-screen flex-col lg:flex-row">
      {/* Mobile top bar (below lg): brand + drawer trigger. From lg up
          the full sidebar below takes over and this bar is hidden. */}
      <div className="flex items-center justify-between gap-3 bg-slate-900 px-4 py-3 text-slate-200 lg:hidden">
        <div className="min-w-0">
          <div className="text-lg font-bold tracking-wide text-white">
            PREDICAI
          </div>
          <div className="text-[11px] text-slate-400">
            Predict Today. Prevent Tomorrow.
          </div>
        </div>
        <button
          onClick={() => setNavOpen(true)}
          aria-label="Open navigation menu"
          className="shrink-0 rounded border border-slate-700 px-2.5 py-2 text-slate-200 transition-colors duration-150 hover:bg-slate-800"
        >
          <MenuIcon />
        </button>
      </div>

      {/* Sidebar (desktop, lg and up) */}
      <aside className="hidden w-60 shrink-0 flex-col bg-slate-900 text-slate-200 lg:flex">
        {sidebarContent}
      </aside>

      {/* Mobile navigation drawer (below lg): the same sidebar content
          in a slide-over panel. Closes on nav selection, backdrop tap,
          the X button, or Escape. */}
      <div
        className={`fixed inset-0 z-50 lg:hidden ${navOpen ? "" : "pointer-events-none"}`}
        aria-hidden={!navOpen}
      >
        <div
          onClick={() => setNavOpen(false)}
          className={`absolute inset-0 bg-slate-950/60 transition-opacity duration-200 motion-reduce:transition-none ${navOpen ? "opacity-100" : "opacity-0"}`}
        />
        <aside
          className={`absolute inset-y-0 left-0 flex w-64 flex-col overflow-y-auto bg-slate-900 text-slate-200 shadow-2xl transition-transform duration-200 ease-out motion-reduce:transition-none ${navOpen ? "translate-x-0" : "-translate-x-full"}`}
        >
          <button
            onClick={() => setNavOpen(false)}
            aria-label="Close navigation menu"
            className="absolute top-4 right-4 rounded border border-slate-700 px-2 py-1.5 text-slate-300 transition-colors duration-150 hover:bg-slate-800"
          >
            <CloseIcon />
          </button>
          {sidebarContent}
        </aside>
      </div>

      {/* Main column */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white px-6 py-3 dark:border-slate-700 dark:bg-slate-900">
          <h1 className="text-base font-bold text-slate-900 sm:text-lg dark:text-slate-100">
            {TITLES[view]}
          </h1>
          <div className="flex items-center gap-3">
            <ApiStatusPill api={api} />
            <button
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              title={
                theme === "dark"
                  ? "Switch to light theme"
                  : "Switch to dark theme"
              }
              aria-label={
                theme === "dark"
                  ? "Switch to light theme"
                  : "Switch to dark theme"
              }
              className="rounded border border-slate-300 bg-white px-2.5 py-1.5 text-slate-600 transition-colors duration-150 hover:border-slate-500 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-300 dark:hover:border-slate-400"
            >
              {theme === "dark" ? <SunIcon /> : <MoonIcon />}
            </button>
          </div>
        </header>

        <main className="flex-1 px-3 py-4 sm:px-6 sm:py-6">
          <div key={view} className="view-enter">
          {view === "fleet" && <FleetView onOpen={openMachine} />}
          {view === "machine" && (
            <MachineView
              engine={engine}
              onSelect={setEngine}
              dark={theme === "dark"}
            />
          )}
          {view === "maintenance" && <MaintenanceView />}
          {view === "model" && <ModelInfoView api={api} />}
          {view === "platform" && <PlatformView />}
          {view === "datasets" && <DatasetsView />}
          {view === "experiments" && <ExperimentsView />}
          {view === "research" && <ResearchView />}
          {view === "privacy" && <PrivacyView />}
          {view === "terms" && <TermsView />}
          </div>
        </main>

        <footer className="border-t border-slate-200 bg-white px-6 py-3 text-xs text-slate-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-400">
          <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
            <span>
              Demo build: FD001 replay data; not connected to real machines.
              · Data exported {data.generated_at} by dashboard/data_export.py
              from simulator/demo.db and ml/artifacts/. · Platform data
              exported {platform.generated_at} by
              dashboard/platform_export.py from platform/experiments/ and
              the dataset registry.
            </span>
            <span className="flex shrink-0 items-center gap-3">
              <button
                onClick={() => setView("privacy")}
                className="font-medium text-slate-600 underline-offset-2 transition-colors duration-150 hover:text-cyan-700 hover:underline dark:text-slate-300 dark:hover:text-cyan-300"
              >
                Privacy Policy
              </button>
              <button
                onClick={() => setView("terms")}
                className="font-medium text-slate-600 underline-offset-2 transition-colors duration-150 hover:text-cyan-700 hover:underline dark:text-slate-300 dark:hover:text-cyan-300"
              >
                Terms and Conditions
              </button>
            </span>
          </div>
        </footer>
      </div>
    </div>
  );
}
