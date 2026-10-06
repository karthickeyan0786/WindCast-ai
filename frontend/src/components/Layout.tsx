import { NavLink, Outlet } from "react-router-dom";
import { useEffect, useState } from "react";
import { api } from "../lib/api";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: "M3 13h8V3H3zM13 21h8V11h-8zM13 3v6h8V3zM3 21h8v-6H3z" },
  { to: "/forecasting", label: "Forecasting", icon: "M3 17l6-6 4 4 8-8M21 7v6h-6" },
  { to: "/analytics", label: "Analytics", icon: "M4 19V5M4 19h16M9 19v-7M14 19v-4M19 19V9" },
  { to: "/model-performance", label: "Model Performance", icon: "M12 20a8 8 0 100-16 8 8 0 000 16zM12 8v4l3 3" },
  { to: "/mlops-monitoring", label: "MLOps Monitoring", icon: "M9 3v18M15 3v18M3 9h18M3 15h18" },
];

function NavIcon({ d }: { d: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
      strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5 shrink-0">
      <path d={d} />
    </svg>
  );
}

export default function Layout() {
  const [modelLoaded, setModelLoaded] = useState<boolean | null>(null);

  useEffect(() => {
    api.health().then((h) => setModelLoaded(h.model_loaded)).catch(() => setModelLoaded(false));
  }, []);

  return (
    <div className="min-h-screen flex bg-background text-on-surface font-body">
      {/* Sidebar */}
      <aside className="w-64 shrink-0 border-r border-outline-variant bg-surface-container-lowest flex flex-col">
        <div className="flex items-center gap-3 px-space-lg py-space-lg">
          <div className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center text-on-primary font-display font-bold">
            W
          </div>
          <div>
            <div className="font-display font-bold text-sm leading-tight">WindCast AI</div>
            <div className="text-[11px] tracking-wide text-on-surface-variant uppercase">
              Energy MLOps Platform
            </div>
          </div>
        </div>

        <nav className="flex-1 px-space-sm space-y-1">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-space-md py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-primary-container text-on-primary-container"
                    : "text-on-surface-variant hover:bg-surface-container"
                }`
              }
            >
              <NavIcon d={item.icon} />
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="px-space-md py-space-lg border-t border-outline-variant">
          <div className="flex items-center gap-2 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                modelLoaded ? "bg-secondary" : "bg-error"
              }`}
            />
            <span className="text-on-surface-variant">
              {modelLoaded === null ? "Checking API…" : modelLoaded ? "Model online" : "API offline"}
            </span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-16 shrink-0 border-b border-outline-variant bg-surface-container-lowest flex items-center justify-between px-space-xl">
          <div className="text-sm text-on-surface-variant">
            Platform <span className="mx-1">/</span>
            <span className="text-on-surface font-medium">Telemetry Console</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="px-3 py-1 rounded-full bg-surface-container text-xs font-mono">
              Model: XGBoost
            </span>
            <div className="w-8 h-8 rounded-full bg-tertiary-container" />
          </div>
        </header>
        <main className="flex-1 overflow-y-auto p-space-xl">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
