import { useEffect, useState } from "react";
import { Card, KpiCard } from "../components/Card";
import { api, DataSummary, HistoryRow, ModelMetadata } from "../lib/api";

function zScore(value: number, mean: number, std: number) {
  return std > 0 ? (value - mean) / std : 0;
}

export default function MlopsMonitoring() {
  const [summary, setSummary] = useState<DataSummary | null>(null);
  const [history, setHistory] = useState<HistoryRow[]>([]);
  const [meta, setMeta] = useState<ModelMetadata | null>(null);

  useEffect(() => {
    api.dataSummary().then(setSummary);
    api.modelMetadata().then(setMeta);
    api.history(100).then(setHistory);
  }, []);

  if (!summary || !meta) return <p className="text-on-surface-variant">Loading monitoring data…</p>;

  const baseline = summary.training_baseline_stats.wind_speed;
  const recentMean =
    history.length > 0
      ? history.reduce((s, h) => s + h.wind_speed, 0) / history.length
      : baseline.mean;
  const drift = zScore(recentMean, baseline.mean, baseline.std);
  const driftStatus = Math.abs(drift) < 1 ? "Stable" : Math.abs(drift) < 2 ? "Watch" : "Drift detected";
  const driftColor =
    driftStatus === "Stable" ? "bg-secondary-container text-on-secondary-container"
    : driftStatus === "Watch" ? "bg-tertiary-container text-on-tertiary-container"
    : "bg-error-container text-on-error-container";

  return (
    <div className="space-y-space-lg">
      <div>
        <h1 className="font-display font-bold text-2xl">MLOps Monitoring</h1>
        <p className="text-sm text-on-surface-variant mt-1">
          Model registry status, live prediction volume, and a data-drift check comparing this
          session's /predict traffic against the training distribution.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-space-md">
        <KpiCard label="Registered Model" value="windcast-power-forecaster" />
        <KpiCard label="Production Version" value={meta.best_model.replace("_", " ")} sub="MLflow Model Registry" />
        <KpiCard label="Predictions this session" value={history.length} />
        <KpiCard
          label="Data Drift Status"
          value={driftStatus}
          sub={`wind_speed z-score: ${drift.toFixed(2)}`}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-space-lg">
        <Card>
          <h2 className="font-display font-semibold text-lg mb-1">Architecture Pipeline</h2>
          <p className="text-sm text-on-surface-variant mb-space-md">
            Weather Data → Data Validation → Feature Engineering → Model Training → MLflow Tracking
            → Model Registry → FastAPI Deployment → Monitoring → Retraining Pipeline
          </p>
          <ol className="space-y-2 text-sm">
            {[
              "Weather Data — raw SCADA CSV / streaming telemetry",
              "Data Validation — schema, range, and null checks (src/data_pipeline.py)",
              "Feature Engineering — hour/day/month + wind_speed³ physics feature",
              "Model Training — LR / RF / XGBoost, TimeSeriesSplit CV (src/train.py)",
              "MLflow Tracking — params, metrics, artifacts per run (./mlflow/mlflow.db)",
              "Model Registry — best run promoted to windcast-power-forecaster",
              "FastAPI Deployment — /predict served from models/best_model.joblib",
              "Monitoring — this page: drift checks + prediction volume",
              "Retraining Pipeline — triggered on schedule or on drift alert",
            ].map((step, i) => (
              <li key={i} className="flex gap-3">
                <span className="w-5 h-5 rounded-full bg-primary text-on-primary text-xs flex items-center justify-center shrink-0">
                  {i + 1}
                </span>
                <span className="text-on-surface-variant">{step}</span>
              </li>
            ))}
          </ol>
        </Card>

        <Card>
          <h2 className="font-display font-semibold text-lg mb-1">Data Drift Check</h2>
          <p className="text-sm text-on-surface-variant mb-space-md">
            Compares the mean of each feature across this session's predictions to the training
            set's mean/std (a simple z-score check — production systems typically use a
            statistical test such as Kolmogorov–Smirnov or Population Stability Index).
          </p>
          <div className={`inline-block px-2.5 py-1 rounded-full text-xs font-mono mb-space-md ${driftColor}`}>
            {driftStatus}
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-on-surface-variant border-b border-outline-variant">
                <th className="py-2 pr-4">Feature</th>
                <th className="py-2 pr-4">Training mean</th>
                <th className="py-2 pr-4">Recent mean</th>
                <th className="py-2 pr-4">z-score</th>
              </tr>
            </thead>
            <tbody>
              {(["wind_speed", "wind_direction", "pressure", "temperature"] as const).map((f) => {
                const b = summary.training_baseline_stats[f];
                const recent = history.length > 0
                  ? history.reduce((s, h) => s + (h as any)[f], 0) / history.length
                  : b.mean;
                const z = zScore(recent, b.mean, b.std);
                return (
                  <tr key={f} className="border-b border-outline-variant last:border-0">
                    <td className="py-2 pr-4 font-medium">{f}</td>
                    <td className="py-2 pr-4">{b.mean.toFixed(2)}</td>
                    <td className="py-2 pr-4">{recent.toFixed(2)}</td>
                    <td className="py-2 pr-4">{z.toFixed(2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {history.length === 0 && (
            <p className="text-xs text-on-surface-variant mt-space-sm">
              No predictions logged yet this session — run some on the Dashboard or Forecasting
              page to populate live drift numbers.
            </p>
          )}
        </Card>

        <Card className="lg:col-span-2">
          <h2 className="font-display font-semibold text-lg mb-1">Retraining Strategy</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-space-md text-sm text-on-surface-variant">
            <div>
              <div className="font-medium text-on-surface mb-1">Scheduled retraining</div>
              Re-run src/train.py on a fixed cadence (e.g. weekly) once new SCADA data has
              accumulated, so the model keeps up with seasonal wind patterns without waiting for
              a problem to appear.
            </div>
            <div>
              <div className="font-medium text-on-surface mb-1">Trigger-based retraining</div>
              Kick off retraining automatically when the drift status above moves to "Drift
              detected", or when live MAE (computed once ground-truth power readings catch up to
              predictions) exceeds a set threshold.
            </div>
            <div>
              <div className="font-medium text-on-surface mb-1">New data ingestion</div>
              Append newly logged SCADA rows to data/, re-run the same data_pipeline.py cleaning
              logic, and retrain — the chronological split naturally extends to include the new
              period.
            </div>
            <div>
              <div className="font-medium text-on-surface mb-1">Rollback strategy</div>
              Every run is versioned in the MLflow Model Registry; if a newly promoted model
              underperforms in production, the previous registry version can be re-served
              immediately without retraining.
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
