"""
WindCast AI — FastAPI Inference Service
==========================================
Loads the trained model ONCE at startup (not per-request — reloading a
1MB joblib file on every call would add needless latency) and exposes:

  POST /predict            -> single prediction
  GET  /health              -> liveness check
  GET  /model/metadata       -> model comparison table + feature importance,
                                 used directly by the frontend's
                                 Model Performance and MLOps Monitoring pages
  GET  /history               -> recent predictions log (in-memory demo store)

Request flow: FastAPI receives JSON -> Pydantic validates types/ranges ->
the SAME feature engineering used in training (wind_speed_cubed, and the
caller-supplied hour/day/month) is assembled into a DataFrame with columns
in the EXACT order the model was trained on -> model.predict() -> the
float is wrapped in a JSON response.

Response flow: Pydantic serialises the response model back to JSON; CORS
middleware is enabled so the React frontend (a different origin during
local dev, e.g. localhost:5173) is allowed to call this API
(localhost:8000) directly from the browser.
"""
import sys, json, time
from pathlib import Path
from datetime import datetime, timezone
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from data_pipeline import FEATURE_COLUMNS, TARGET, load_dataset  # single source of truth

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
DATA_PATH = ROOT / "data" / "TexasTurbine.csv"

app = FastAPI(
    title="WindCast AI",
    description="Intelligent Renewable Energy Forecasting API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the deployed frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model loading (once, at import time) ---------------------------------
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. Run `python src/train.py` first."
    )
model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text())

# Load the cleaned dataset once so the Analytics/MLOps Monitoring pages
# can be driven by real numbers instead of hardcoded frontend placeholders.
_df = load_dataset(DATA_PATH)
_train_df = _df.iloc[: metadata["train_rows"]]  # matches train.py's chronological split

# In-memory prediction log for the "Predictions History" dashboard page.
# In a real deployment this would be a database table, not a Python list.
_prediction_log: List[dict] = []


class PredictRequest(BaseModel):
    wind_speed: float = Field(..., ge=0, le=40, description="Wind speed in m/s")
    wind_direction: float = Field(..., ge=0, le=360, description="Wind direction in degrees")
    pressure: float = Field(..., ge=0.8, le=1.2, description="Atmospheric pressure in atm")
    temperature: float = Field(..., ge=-30, le=55, description="Air temperature in Celsius")
    hour: int = Field(..., ge=0, le=23)
    day: int = Field(..., ge=1, le=31)
    month: int = Field(..., ge=1, le=12)

    class Config:
        json_schema_extra = {
            "example": {
                "wind_speed": 9.5, "wind_direction": 141, "pressure": 0.999,
                "temperature": 20.5, "hour": 12, "day": 15, "month": 6,
            }
        }


class PredictResponse(BaseModel):
    predicted_power_kw: float
    model_used: str
    timestamp: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    row = req.model_dump()
    row["wind_speed_cubed"] = row["wind_speed"] ** 3  # same feature engineering as training
    X = pd.DataFrame([row])[FEATURE_COLUMNS]  # enforce training-time column order

    try:
        pred = float(model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    pred = max(0.0, pred)  # power can't be negative
    result = {
        "predicted_power_kw": round(pred, 2),
        "model_used": metadata["best_model"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _prediction_log.append({**row, **result})
    if len(_prediction_log) > 500:
        _prediction_log.pop(0)
    return result


@app.get("/data/summary")
def data_summary():
    """
    Powers the Analytics page and the MLOps Monitoring drift baseline.
    Computed from the real cleaned dataset (see src/data_pipeline.py) —
    not synthetic frontend placeholders.
    """
    cols = ["wind_speed", "wind_direction", "pressure", "temperature", TARGET]
    corr = _df[cols].corr().round(3)

    def histogram(col: str, bins: int = 20):
        counts, edges = pd.cut(_df[col], bins=bins, retbins=True)
        vc = counts.value_counts(sort=False)
        return {
            "bin_edges": [round(float(e), 2) for e in edges],
            "counts": [int(c) for c in vc.values],
        }

    # Subsample the scatter so the payload stays small (~300 points is
    # plenty to show the turbine power-curve shape on a chart).
    sample = _df.sample(n=min(300, len(_df)), random_state=42).sort_values("wind_speed")

    return {
        "row_count": len(_df),
        "correlation": {c: corr[c].to_dict() for c in cols},
        "histograms": {c: histogram(c) for c in cols},
        "scatter_wind_vs_power": [
            {"wind_speed": round(float(r.wind_speed), 2), "power_kw": round(float(r.power_kw), 1)}
            for r in sample.itertuples()
        ],
        "training_baseline_stats": {
            c: {"mean": float(_train_df[c].mean()), "std": float(_train_df[c].std())}
            for c in ["wind_speed", "wind_direction", "pressure", "temperature", TARGET]
        },
    }


@app.get("/model/metadata")
def model_metadata():
    """Powers the Model Performance & MLOps Monitoring dashboard pages."""
    return metadata


@app.get("/history")
def history(limit: int = 50):
    """Powers the Predictions History Log dashboard page."""
    return list(reversed(_prediction_log[-limit:]))


@app.get("/")
def root():
    return {
        "service": "WindCast AI Inference API",
        "docs": "/docs",
        "endpoints": ["/predict", "/health", "/model/metadata", "/history"],
    }
