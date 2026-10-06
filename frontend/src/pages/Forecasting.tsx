import { useState } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from "recharts";
import { Card } from "../components/Card";
import { api } from "../lib/api";

const BASE = { wind_direction: 140, pressure: 0.999, temperature: 20, hour: 12, day: 15, month: 6 };

export default function Forecasting() {
  const [curve, setCurve] = useState<{ wind_speed: number; predicted_power_kw: number }[]>([]);
  const [loading, setLoading] = useState(false);
  const [scenario, setScenario] = useState(BASE);

  const runSweep = async () => {
    setLoading(true);
    try {
      const speeds = Array.from({ length: 21 }, (_, i) => i); // 0..20 m/s
      const results = await Promise.all(
        speeds.map((ws) =>
          api.predict({ ...scenario, wind_speed: ws }).then((r) => ({
            wind_speed: ws,
            predicted_power_kw: r.predicted_power_kw,
          }))
        )
      );
      setCurve(results);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-space-lg">
      <div>
        <h1 className="font-display font-bold text-2xl">Forecasting Engine</h1>
        <p className="text-sm text-on-surface-variant mt-1">
          Simulate the model's predicted power curve across a full wind-speed sweep for a fixed
          atmospheric scenario — the classic way to sanity-check a turbine power model against
          physical expectations (cut-in, rise, and rated-power plateau).
        </p>
      </div>

      <Card>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-space-sm mb-space-md">
          <ScenarioField label="Direction (°)" value={scenario.wind_direction}
            onChange={(v) => setScenario({ ...scenario, wind_direction: v })} />
          <ScenarioField label="Pressure (atm)" value={scenario.pressure} step={0.001}
            onChange={(v) => setScenario({ ...scenario, pressure: v })} />
          <ScenarioField label="Temp (°C)" value={scenario.temperature}
            onChange={(v) => setScenario({ ...scenario, temperature: v })} />
          <ScenarioField label="Hour" value={scenario.hour}
            onChange={(v) => setScenario({ ...scenario, hour: v })} />
          <ScenarioField label="Month" value={scenario.month}
            onChange={(v) => setScenario({ ...scenario, month: v })} />
        </div>
        <button
          onClick={runSweep}
          disabled={loading}
          className="bg-primary text-on-primary text-sm font-medium px-4 py-2 rounded-lg disabled:opacity-50"
        >
          {loading ? "Calling /predict 21 times…" : "Run Power Curve Simulation"}
        </button>
      </Card>

      <Card>
        <h2 className="font-display font-semibold text-lg mb-1">AI-Predicted Power Curve</h2>
        <p className="text-sm text-on-surface-variant mb-space-md">
          Predicted output (kW) vs. wind speed (m/s), all other conditions held fixed.
        </p>
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={curve}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e7ff" />
            <XAxis dataKey="wind_speed" label={{ value: "Wind speed (m/s)", position: "insideBottom", offset: -5, fontSize: 12 }} />
            <YAxis label={{ value: "Predicted power (kW)", angle: -90, position: "insideLeft", fontSize: 12 }} />
            <Tooltip formatter={(v: number) => [`${v} kW`, "Predicted"]} labelFormatter={(l) => `${l} m/s`} />
            <Legend />
            <Line type="monotone" dataKey="predicted_power_kw" name="Predicted Power Output (AI Model kW)"
              stroke="#006194" strokeWidth={2.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
        {curve.length === 0 && (
          <p className="text-sm text-on-surface-variant mt-2">
            Run the simulation to see the model's learned power curve for this scenario.
          </p>
        )}
      </Card>
    </div>
  );
}

function ScenarioField({
  label, value, onChange, step = 1,
}: { label: string; value: number; onChange: (v: number) => void; step?: number }) {
  return (
    <label className="text-xs text-on-surface-variant">
      {label}
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="mt-1 w-full border border-outline-variant rounded-lg px-2 py-1.5 text-sm bg-surface-container-lowest focus:outline-none focus:ring-2 focus:ring-primary"
      />
    </label>
  );
}
