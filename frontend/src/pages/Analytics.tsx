import { useEffect, useState } from "react";
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  BarChart, Bar,
} from "recharts";
import { Card } from "../components/Card";
import { api, DataSummary } from "../lib/api";

const FEATURES = ["wind_speed", "wind_direction", "pressure", "temperature", "power_kw"];

function corrColor(v: number) {
  // Diverging scale: negative -> tertiary hue, 0 -> neutral, positive -> primary hue
  const a = Math.abs(v);
  if (v >= 0) return `rgba(0, 97, 148, ${0.15 + a * 0.75})`;
  return `rgba(186, 26, 26, ${0.15 + a * 0.75})`;
}

export default function Analytics() {
  const [summary, setSummary] = useState<DataSummary | null>(null);

  useEffect(() => {
    api.dataSummary().then(setSummary);
  }, []);

  if (!summary) {
    return <p className="text-on-surface-variant">Loading dataset analytics…</p>;
  }

  const windHist = summary.histograms.wind_speed.counts.map((c, i) => ({
    bin: summary.histograms.wind_speed.bin_edges[i].toFixed(1),
    count: c,
  }));
  const powerHist = summary.histograms.power_kw.counts.map((c, i) => ({
    bin: summary.histograms.power_kw.bin_edges[i].toFixed(0),
    count: c,
  }));

  return (
    <div className="space-y-space-lg">
      <div>
        <h1 className="font-display font-bold text-2xl">Analytics &amp; Visualizations</h1>
        <p className="text-sm text-on-surface-variant mt-1">
          Computed live from the cleaned dataset ({summary.row_count.toLocaleString()} hourly readings) —
          the same figures used to justify feature and model choices in a project review.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-space-lg">
        <Card>
          <h2 className="font-display font-semibold text-lg mb-1">Correlation Heatmap</h2>
          <p className="text-sm text-on-surface-variant mb-space-md">
            wind_speed correlates with power at r ={" "}
            {summary.correlation.wind_speed.power_kw.toFixed(2)} — the dominant linear relationship.
          </p>
          <div className="overflow-x-auto">
            <table className="text-xs border-collapse">
              <thead>
                <tr>
                  <th className="p-1" />
                  {FEATURES.map((f) => (
                    <th key={f} className="p-1 font-medium text-on-surface-variant">{f}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {FEATURES.map((row) => (
                  <tr key={row}>
                    <td className="p-1 font-medium text-on-surface-variant text-right pr-2">{row}</td>
                    {FEATURES.map((col) => (
                      <td
                        key={col}
                        className="p-1 text-center rounded"
                        style={{ background: corrColor(summary.correlation[row][col]) }}
                      >
                        {summary.correlation[row][col].toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card>
          <h2 className="font-display font-semibold text-lg mb-1">Wind Speed vs. Power Output</h2>
          <p className="text-sm text-on-surface-variant mb-space-md">
            The classic S-shaped turbine power curve — near-zero below cut-in speed, a steep rise,
            then a rated-power plateau.
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e7ff" />
              <XAxis dataKey="wind_speed" name="Wind speed" unit=" m/s" tick={{ fontSize: 11 }} />
              <YAxis dataKey="power_kw" name="Power" unit=" kW" tick={{ fontSize: 11 }} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} />
              <Scatter data={summary.scatter_wind_vs_power} fill="#006c4a" fillOpacity={0.5} />
            </ScatterChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h2 className="font-display font-semibold text-lg mb-1">Wind Speed Distribution</h2>
          <p className="text-sm text-on-surface-variant mb-space-md">
            Roughly bell-shaped (Weibull-like), centered near{" "}
            {summary.training_baseline_stats.wind_speed.mean.toFixed(1)} m/s — typical for
            site-level wind data.
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={windHist}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e7ff" />
              <XAxis dataKey="bin" tick={{ fontSize: 10 }} interval={2} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#006194" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h2 className="font-display font-semibold text-lg mb-1">Power Output Distribution</h2>
          <p className="text-sm text-on-surface-variant mb-space-md">
            Right-skewed with a mass near 0 kW — calm hours where output is near zero are common.
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={powerHist}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e7ff" />
              <XAxis dataKey="bin" tick={{ fontSize: 10 }} interval={2} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#0051d5" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}
