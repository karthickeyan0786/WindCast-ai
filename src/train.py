"""
WindCast AI — Model Training
==============================
Trains Linear Regression (baseline), Random Forest, and XGBoost on a
CHRONOLOGICAL split, cross-validates with TimeSeriesSplit, logs every run
to MLflow (params, metrics, artifacts, and registers the best model), and
saves the winning model + metadata to models/ for the FastAPI service.

Run: python src/train.py
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parent))
from data_pipeline import load_dataset, get_X_y, FEATURE_COLUMNS, TARGET

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow' / 'mlflow.db'}")
mlflow.set_experiment("windcast-ai-power-forecasting")


def chronological_split(X: pd.DataFrame, y: pd.Series, test_frac: float = 0.2):
    """
    WHY NOT random train_test_split:
    A random split lets rows from the SAME night or the SAME storm system
    land in both train and test. Because wind speed changes smoothly hour
    to hour, a training row at 2:00am and a test row at 3:00am on the same
    night are highly correlated — the model isn't really being tested on
    unseen conditions, it's echoing its neighbour. That inflates the
    reported R²/MAE and is a textbook case of DATA LEAKAGE in time series.

    The fix: split chronologically — train on the earlier months, test on
    the later, untouched months, which mimics how the model will actually
    be used in production (forecasting the future from the past).
    """
    n = len(X)
    split_idx = int(n * (1 - test_frac))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def time_series_cv_score(model, X_train, y_train, n_splits=5) -> dict:
    """
    TimeSeriesSplit: instead of one arbitrary train/test cut, it makes
    n_splits FORWARD-CHAINING folds — fold 1 trains on the earliest chunk
    and validates on the next chunk, fold 2 trains on the first two chunks
    and validates on the third, and so on. Every validation fold is always
    strictly AFTER its training data, so no fold ever leaks future
    information backward. This gives a more robust estimate of real-world
    generalisation than a single split, without ever shuffling time order.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    for train_idx, val_idx in tscv.split(X_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        pred = model.predict(X_train.iloc[val_idx])
        maes.append(mean_absolute_error(y_train.iloc[val_idx], pred))
    return {"cv_mae_mean": float(np.mean(maes)), "cv_mae_std": float(np.std(maes))}


def main():
    df = load_dataset(ROOT / "data" / "TexasTurbine.csv")
    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = chronological_split(X, y, test_frac=0.2)
    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}  "
          f"(chronological — test set is the final {len(X_test)/len(X)*100:.0f}% of the year)")

    models = {
        "linear_regression": (
            LinearRegression(),
            {},
        ),
        "random_forest": (
            RandomForestRegressor(
                n_estimators=300,      # more trees = lower variance, diminishing returns past ~300
                max_depth=14,          # caps tree depth to reduce overfitting on 8.7k rows
                min_samples_leaf=3,    # each leaf needs >=3 samples -> smoother predictions
                n_jobs=-1,
                random_state=42,
            ),
            {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 3},
        ),
        "xgboost": (
            XGBRegressor(
                n_estimators=400,      # boosting rounds
                learning_rate=0.05,    # small step size -> needs more rounds but generalises better
                max_depth=5,           # shallow trees per round (boosting builds strength via many weak learners)
                subsample=0.8,         # row subsampling -> regularization, reduces overfitting
                colsample_bytree=0.8,  # feature subsampling -> regularization, decorrelates trees
                reg_lambda=1.0,        # L2 regularization on leaf weights
                random_state=42,
                n_jobs=-1,
            ),
            {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 5,
             "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        ),
    }

    results = []
    fitted_models = {}

    for name, (model, params) in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            mlflow.set_tag("model_family", name)
            mlflow.set_tag("split_strategy", "chronological_80_20")

            cv_scores = time_series_cv_score(model, X_train, y_train, n_splits=5)
            mlflow.log_metrics(cv_scores)

            # Final fit on the FULL training set, evaluated on the held-out test set
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = evaluate(y_test, preds)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(model, artifact_path="model", serialization_format="pickle")

            fitted_models[name] = model
            results.append({"Model": name, **cv_scores, **metrics})
            print(f"[{name}] CV MAE={cv_scores['cv_mae_mean']:.2f} | "
                  f"Test MAE={metrics['MAE']:.2f} RMSE={metrics['RMSE']:.2f} R2={metrics['R2']:.4f}")

    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("\n=== Model Comparison (sorted by R2) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    print(f"\nSelected best model: {best_name}")

    # Register the winner in the MLflow Model Registry as "windcast-power-forecaster"
    with mlflow.start_run(run_name=f"{best_name}_registered"):
        mlflow.log_metrics(results_df.iloc[0][["MAE", "RMSE", "R2"]].to_dict())
        model_info = mlflow.sklearn.log_model(
            best_model, artifact_path="model", serialization_format="pickle",
            registered_model_name="windcast-power-forecaster",
        )

    # Save for the FastAPI service (simple joblib load, no MLflow dependency at inference)
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    # Feature importance (RF / XGBoost only — Linear Regression uses coefficients)
    importance_records = []
    if hasattr(best_model, "feature_importances_"):
        for feat, imp in zip(FEATURE_COLUMNS, best_model.feature_importances_):
            importance_records.append({"feature": feat, "importance": float(imp)})
        importance_records.sort(key=lambda r: -r["importance"])
    elif hasattr(best_model, "coef_"):
        for feat, coef in zip(FEATURE_COLUMNS, best_model.coef_):
            importance_records.append({"feature": feat, "importance": float(abs(coef))})
        importance_records.sort(key=lambda r: -r["importance"])

    metadata = {
        "best_model": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "test_metrics": results_df.iloc[0][["MAE", "RMSE", "R2"]].to_dict(),
        "comparison_table": results_df.to_dict(orient="records"),
        "feature_importance": importance_records,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved model -> models/best_model.joblib")
    print("Saved metadata -> models/model_metadata.json")
    print("MLflow runs stored under ./mlflow/mlflow.db  "
          "(run `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db` to view)")


if __name__ == "__main__":
    main()
