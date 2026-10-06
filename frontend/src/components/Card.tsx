import { ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`bg-surface-container-lowest border border-outline-variant rounded-xl p-space-lg ${className}`}
    >
      {children}
    </div>
  );
}

export function KpiCard({
  label,
  value,
  unit,
  sub,
  trend,
}: {
  label: string;
  value: string | number;
  unit?: string;
  sub?: string;
  trend?: { value: string; positive: boolean };
}) {
  return (
    <Card>
      <div className="text-xs font-medium text-on-surface-variant uppercase tracking-wide mb-2">
        {label}
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-2xl font-display font-bold">{value}</span>
        {unit && <span className="text-sm text-on-surface-variant">{unit}</span>}
      </div>
      <div className="flex items-center gap-2 mt-2 text-xs">
        {sub && <span className="text-on-surface-variant">{sub}</span>}
        {trend && (
          <span
            className={`px-1.5 py-0.5 rounded-full font-mono ${
              trend.positive
                ? "bg-secondary-container text-on-secondary-container"
                : "bg-error-container text-on-error-container"
            }`}
          >
            {trend.positive ? "▲" : "▼"} {trend.value}
          </span>
        )}
      </div>
    </Card>
  );
}
