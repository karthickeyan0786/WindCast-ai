"""
WindCast AI — Data Pipeline
============================
Loads the raw Texas turbine SCADA-style dataset, cleans it, and engineers
the features used by every downstream model.

Why this file is separate from train.py:
In a real MLOps setup, the SAME cleaning logic must run identically at
training time and at inference time (inside the FastAPI service). Keeping
it in one importable module (instead of copy-pasted notebook cells) is
what prevents "training/serving skew" — a classic production ML bug where
the API preprocesses a request slightly differently than training did,
silently corrupting predictions.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

RAW_COLUMNS = {
    "Time stamp": "timestamp",
    "System power generated | (kW)": "power_kw",
    "Wind speed | (m/s)": "wind_speed",
    "Wind direction | (deg)": "wind_direction",
    "Pressure | (atm)": "pressure",
    "Air temperature | ('C)": "temperature",
}

TARGET = "power_kw"

# Final feature order — the FastAPI request schema and the trained model
# must agree on this exact order, so it is defined once, here.
FEATURE_COLUMNS = [
    "wind_speed",
    "wind_direction",
    "pressure",
    "temperature",
    "hour",
    "day",
    "month",
    "wind_speed_cubed",   # physics-informed feature, see note below
]


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and rename columns to clean, code-friendly names."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns=RAW_COLUMNS)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning strategy (and why each step exists):

    1. Duplicate removal — SCADA loggers occasionally double-write a
       timestamp on reconnect. Duplicates would let the model "memorize"
       a row instead of generalising, so they are dropped.
    2. Timestamp parsing — the raw column is a string like
       "Jan 1, 12:00 am". Parsing it to datetime is required before we
       can derive hour/day/month, and before we can sort chronologically
       for the time-series split later.
    3. Missing values — turbines can report NaN during sensor faults or
       curtailment events. We do NOT silently drop these rows, because
       for an hourly time series that creates a "hole" that would corrupt
       a chronological split. Instead we linearly interpolate the small
       gaps (physically justified: wind speed/temperature change
       smoothly hour to hour) and only drop rows where the TARGET itself
       is missing (a synthetic power value would be invented data).
    4. Outlier handling — physically impossible values (negative power,
       negative wind speed, pressure far outside the ~0.9-1.1 atm range
       for this site) are clipped rather than deleted, so we don't lose
       otherwise-valid neighbouring readings; a small IQR-based clip is
       also applied to the target to reduce the influence of a handful
       of extreme sensor spikes without discarding real data.
    """
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    dup_removed = before - len(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%b %d, %I:%M %p")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Drop rows with a missing target — we cannot manufacture ground truth.
    df = df.dropna(subset=[TARGET])

    # Interpolate small sensor gaps in the predictors only.
    predictor_cols = ["wind_speed", "wind_direction", "pressure", "temperature"]
    df[predictor_cols] = df[predictor_cols].interpolate(method="linear", limit=6)
    df = df.dropna(subset=predictor_cols)  # drop any gap too long to interpolate safely

    # Physically impossible values -> clip, don't delete.
    df["wind_speed"] = df["wind_speed"].clip(lower=0)
    df["power_kw"] = df["power_kw"].clip(lower=0)
    df["pressure"] = df["pressure"].clip(lower=0.9, upper=1.1)
    df["wind_direction"] = df["wind_direction"] % 360

    # IQR clip on the target to tame extreme sensor spikes (1.5x IQR rule).
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    iqr = q3 - q1
    df[TARGET] = df[TARGET].clip(lower=max(0, q1 - 1.5 * iqr), upper=q3 + 1.5 * iqr)

    df.attrs["duplicates_removed"] = dup_removed
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering — why each new feature exists:

    - hour / day / month: extracted from the timestamp so the model can
      learn diurnal and seasonal wind patterns (wind is typically stronger
      at night/early morning and seasonally stronger in certain months).
      Trees (RF/XGBoost) handle these as ordinary numeric splits; we keep
      them as plain integers rather than one-hot encoding because that
      keeps the feature space small and trees can already split on
      thresholds (e.g. hour < 6) to recover cyclic structure.
    - wind_speed_cubed: turbine physics — power in wind is proportional to
      velocity CUBED (P = 0.5 * air_density * swept_area * v^3, before the
      turbine's own efficiency curve/cut-in/cut-out limits apply). Giving
      the model v^3 directly as a feature lets even the LINEAR baseline
      capture this non-linearity, and it consistently shows up as a top
      feature for the tree models too.
    """
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["wind_speed_cubed"] = df["wind_speed"] ** 3
    return df


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Full pipeline: load -> clean -> engineer. This is the single function
    both train.py and the FastAPI service should treat as the source of
    truth for how raw data becomes model-ready data."""
    df = load_raw(csv_path)
    df = clean(df)
    df = engineer_features(df)
    return df


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    return X, y


if __name__ == "__main__":
    df = load_dataset(Path(__file__).resolve().parents[1] / "data" / "TexasTurbine.csv")
    print(df.shape)
    print(df[FEATURE_COLUMNS + [TARGET]].describe())
