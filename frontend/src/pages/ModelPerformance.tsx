import { useEffect, useState } from "react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell,
} from "recharts";
import { Card, KpiCard } from "../components/Card";
import { api, ModelMetadata } from "../lib/api";

const MODEL_COLORS: Record<string, string> = {
  linear_regression: "#707881",
  random_forest: "#006c4a",
  xgboost: "#006194",
};

export default function ModelPerformance() {
  const [meta, setMeta] = useState<ModelMetadata | null>(null);

  useEffect(() => {
    api.modelMetadata().then(setMeta);
  }, []);

  if (!meta) return <p className="text-on-surface-variant">Loading model registry…</p>;

  const maeData = meta.comparison_table.map((r) => ({ model: r.Model, MAE: +r.MAE.toFixed(1) }));
  const r2Data = meta.comparison_table.map((r) => ({ model: r.Model, R2: +r.R2.toFixed(4) }));
  const importanceData = meta.feature_importance.map((f) => ({
    feature: f.feature,
    importance: +(f.importance * 100).toFixed(2),
  }));

  return (
    <div className="space-y-space-lg">
      <div>
        <h1 className="font-display font-bold text-2xl">Model Performance Benchmark</h1>
        <p className="text-sm text-on-surface-variant mt-1">
          Chronological hold-out test set ({meta.test_rows.toLocaleString()} rows) — the final
          20% of the year, never seen during training or cross-validation.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-space-md">
        <KpiCard label="Selected Model" value={meta.best_model.replace("_", " ")} />
        <KpiCard label="Test MAE" value={meta.test_metrics.MAE.toFixed(2)} unit="kW" />
        <KpiCard label="Test RMSE" value={meta.test_metrics.RMSE.toFixed(2)} unit="kW" />
        <KpiCard label="Test R²" value={meta.test_metrics.R2.toFixed(4)} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-space-lg">
        <Card>
          <h2 className="font-display font-semibold text-lg mb-1">Mean Absolute Error by Model</h2>
          <p className="text-sm text-on-surface-variant mb-space-md">
            Lower is better — average kW the model's prediction is off by. Linear Regression sets
            the baseline; the ensembles cut error by over 20x by capturing the non-linear power curve.
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={maeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e7ff" />
              <XAxis dataKey="model" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} unit=" kW" />
              <Tooltip formatter={(v: number) => [`${v} kW`, "MAE"]} />
              <Bar dataKey="MAE" radius={[4, 4, 0, 0]}>
                {maeData.map((d) => (
                  <Cell key={d.model} fill={MODEL_COLORS[d.model] ?? "#006194"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h2 className="font-display font-semibold text-lg mb-1">R² Score by Model</h2>
          <p className="text-sm text-on-surface-variant mb-space-md">
            Fraction of variance in power output explained by the model (1.0 = perfect).
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={r2Data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e7ff" />
              <XAxis dataKey="model" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} />
              <Tooltip />
              <Bar dataKey="R2" radius={[4, 4, 0, 0]}>
                {r2Data.map((d) => (
                  <Cell key={d.model} fill={MODEL_COLORS[d.model] ?? "#006194"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card className="lg:col-span-2">
          <h2 className="font-display font-semibold text-lg mb-1">
            Feature Importance — {meta.best_model.replace("_", " ")}
          </h2>
          <p className="text-sm text-on-surface-variant mb-space-md">
            wind_speed and its cubed transform dominate — consistent with turbine physics
            (P ∝ v³) and with the correlation heatmap on the Analytics page.
          </p>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={importanceData} layout="vertical" margin={{ left: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e7ff" />
              <XAxis type="number" unit="%" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={110} />
              <Tooltip formatter={(v: number) => [`${v}%`, "Importance"]} />
              <Bar dataKey="importance" fill="#006194" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>

      <Card>
        <h2 className="font-display font-semibold text-lg mb-space-md">Full Comparison Table</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-on-surface-variant border-b border-outline-variant">
                <th className="py-2 pr-4">Model</th>
                <th className="py-2 pr-4">CV MAE (mean)</th>
                <th className="py-2 pr-4">CV MAE (std)</th>
                <th className="py-2 pr-4">Test MAE</th>
                <th className="py-2 pr-4">Test RMSE</th>
                <th className="py-2 pr-4">Test R²</th>
              </tr>
            </thead>
            <tbody>
              {meta.comparison_table.map((r) => (
                <tr key={r.Model} className="border-b border-outline-variant last:border-0">
                  <td className="py-2 pr-4 font-medium">
                    {r.Model === meta.best_model && (
                      <span className="inline-block w-2 h-2 rounded-full bg-secondary mr-2" />
                    )}
                    {r.Model}
                  </td>
                  <td className="py-2 pr-4">{r.cv_mae_mean.toFixed(2)}</td>
                  <td className="py-2 pr-4">{r.cv_mae_std.toFixed(2)}</td>
                  <td className="py-2 pr-4">{r.MAE.toFixed(2)}</td>
                  <td className="py-2 pr-4">{r.RMSE.toFixed(2)}</td>
                  <td className="py-2 pr-4">{r.R2.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
