// Thin typed wrapper around the WindCast AI FastAPI backend.
// All frontend network calls go through here so the request/response
// shape only needs to match the backend's Pydantic models in one place.

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

export interface PredictRequest {
  wind_speed: number;
  wind_direction: number;
  pressure: number;
  temperature: number;
  hour: number;
  day: number;
  month: number;
}

export interface PredictResponse {
  predicted_power_kw: number;
  model_used: string;
  timestamp: string;
}

export interface ModelComparisonRow {
  Model: string;
  cv_mae_mean: number;
  cv_mae_std: number;
  MAE: number;
  RMSE: number;
  R2: number;
}

export interface FeatureImportanceRow {
  feature: string;
  importance: number;
}

export interface ModelMetadata {
  best_model: string;
  feature_columns: string[];
  target: string;
  test_metrics: { MAE: number; RMSE: number; R2: number };
  comparison_table: ModelComparisonRow[];
  feature_importance: FeatureImportanceRow[];
  train_rows: number;
  test_rows: number;
}

export type HistoryRow = PredictRequest & PredictResponse & { wind_speed_cubed: number };

export interface DataSummary {
  row_count: number;
  correlation: Record<string, Record<string, number>>;
  histograms: Record<string, { bin_edges: number[]; counts: number[] }>;
  scatter_wind_vs_power: { wind_speed: number; power_kw: number }[];
  training_baseline_stats: Record<string, { mean: number; std: number }>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${path} failed: ${res.status} ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string; model_loaded: boolean }>("/health"),
  predict: (payload: PredictRequest) =>
    request<PredictResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  modelMetadata: () => request<ModelMetadata>("/model/metadata"),
  history: (limit = 50) => request<HistoryRow[]>(`/history?limit=${limit}`),
  dataSummary: () => request<DataSummary>("/data/summary"),
};
