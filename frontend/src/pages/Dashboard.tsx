import { useEffect, useState } from "react";
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
} from "recharts";
import { Card, KpiCard } from "../components/Card";
import { api, HistoryRow, ModelMetadata } from "../lib/api";

const DEFAULT_INPUT = {
  wind_speed: 9.4, wind_direction: 141, pressure: 0.999,
  temperature: 20.5, hour: new Date().getHours(), day: 15, month: 6,
};

export default function Dashboard() {
  const [input, setInput] = useState(DEFAULT_INPUT);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [history, setHistory] = useState<HistoryRow[]>([]);
  const [meta, setMeta] = useState<ModelMetadata | null>(null);
  const [loading, setLoading] = useState(false);

  const refreshHistory = () => api.history(24).then((h) => setHistory(h.reverse()));

  useEffect(() => {
    api.modelMetadata().then(setMeta);
    refreshHistory();
  }, []);

  const runPrediction = async () => {
    setLoading(true);
    try {
      const res = await api.predict(input);
      setPrediction(res.predicted_power_kw);
      await refreshHistory();
    } finally {
      setLoading(false);
    }
  };

  const chartData = history.map((h, i) => ({
    idx: i + 1,
    power: h.predicted_power_kw,
  }));

  return (
    <div className="space-y-space-lg">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="font-display font-bold text-2xl">Renewable Energy Operations Dashboard</h1>
          {meta && (
            <span className="inline-block mt-1 px-2.5 py-0.5 rounded-full bg-secondary-container text-on-secondary-container text-xs font-mono">
              Model: {meta.best_model.toUpperCase()} · R² {meta.test_metrics.R2.toFixed(4)}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-space-md">
        <KpiCard label="Wind Velocity" value={input.wind_speed} unit="m/s" sub="Hub height" />
        <KpiCard label="Ambient Temp" value={input.temperature} unit="°C" sub="Nacelle sensor" />
        <KpiCard label="Barometric Pressure" value={input.pressure} unit="atm" sub="Stable front" />
        <KpiCard
          label="AI Predicted Output"
          value={prediction !== null ? prediction.toLocaleString() : "—"}
          unit="kW"
          sub={meta ? `Model MAE ±${meta.test_metrics.MAE.toFixed(1)} kW` : undefined}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-space-lg">
        <Card className="lg:col-span-2">
          <div className="flex items-center justify-between mb-space-md">
            <div>
              <h2 className="font-display font-semibold text-lg">Recent Predictions Trend</h2>
              <p className="text-sm text-on-surface-variant">
                Output of the last {history.length} calls to the /predict endpoint this session.
              </p>
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="powerFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#006194" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#006194" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e7ff" />
              <XAxis dataKey="idx" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} unit=" kW" />
              <Tooltip formatter={(v: number) => [`${v} kW`, "Predicted power"]} />
              <Area type="monotone" dataKey="power" stroke="#006194" fill="url(#powerFill)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
          {history.length === 0 && (
            <p className="text-sm text-on-surface-variant mt-2">
              Run a prediction on the right to start building this trend.
            </p>
          )}
        </Card>

        <Card>
          <h2 className="font-display font-semibold text-lg mb-space-md">Run Manual Prediction</h2>
          <div className="grid grid-cols-2 gap-space-sm">
            <Field label="Wind speed (m/s)" value={input.wind_speed}
              onChange={(v) => setInput({ ...input, wind_speed: v })} step={0.1} />
            <Field label="Direction (°)" value={input.wind_direction}
              onChange={(v) => setInput({ ...input, wind_direction: v })} step={1} />
            <Field label="Pressure (atm)" value={input.pressure}
              onChange={(v) => setInput({ ...input, pressure: v })} step={0.001} />
            <Field label="Temp (°C)" value={input.temperature}
              onChange={(v) => setInput({ ...input, temperature: v })} step={0.1} />
            <Field label="Hour" value={input.hour}
              onChange={(v) => setInput({ ...input, hour: v })} step={1} />
            <Field label="Month" value={input.month}
              onChange={(v) => setInput({ ...input, month: v })} step={1} />
          </div>
          <button
            onClick={runPrediction}
            disabled={loading}
            className="mt-space-md w-full bg-primary text-on-primary font-medium text-sm py-2.5 rounded-lg hover:opacity-90 disabled:opacity-50 transition-opacity"
          >
            {loading ? "Running inference…" : "Run AI Forecast"}
          </button>
          {prediction !== null && (
            <div className="mt-space-md text-center">
              <div className="text-xs text-on-surface-variant uppercase tracking-wide">Predicted Output</div>
              <div className="text-3xl font-display font-bold text-primary">
                {prediction.toLocaleString()} <span className="text-base">kW</span>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

function Field({
  label, value, onChange, step,
}: { label: string; value: number; onChange: (v: number) => void; step: number }) {
  return (
    <label className="text-xs text-on-surface-variant">
      {label}
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="mt-1 w-full border border-outline-variant rounded-lg px-2 py-1.5 text-sm text-on-surface bg-surface-container-lowest focus:outline-none focus:ring-2 focus:ring-primary"
      />
    </label>
  );
}
