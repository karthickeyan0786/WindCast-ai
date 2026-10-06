# WindCast AI
### Intelligent Renewable Energy Forecasting & MLOps-Based Smart Power Optimization Framework

A production-style ML system that forecasts wind turbine power output from
weather conditions, with full MLOps tracking (MLflow), a FastAPI inference
service, and a React + TypeScript operations dashboard.

Read `docs/CONCEPTS_GUIDE.md` for the "why" behind every design decision,
and `docs/VIVA_QUESTIONS.md` for review/interview prep.

## Project Structure

```
WindCast_AI/
├── data/TexasTurbine.csv         Raw dataset (8,760 hourly readings)
├── src/
│   ├── data_pipeline.py           Cleaning + feature engineering (shared by training & API)
│   ├── eda.py                    Generates EDA figures -> outputs/figures/
│   └── train.py                    Trains LR/RF/XGBoost, TimeSeriesSplit CV, MLflow tracking
├── models/                     best_model.joblib + model_metadata.json (produced by train.py)
├── api/
│   ├── main.py                   FastAPI service (/predict, /health, /model/metadata, /history, /data/summary)
│   └── requirements.txt
├── frontend/                   React + TypeScript dashboard (Vite + Tailwind)
│   └── src/pages/                Dashboard, Forecasting, Analytics, ModelPerformance, MlopsMonitoring
├── mlflow/                     MLflow sqlite tracking store (created by train.py)
├── docker/Dockerfile             Container build for the API
├── outputs/                    EDA figures + model comparison report
└── docs/                       CONCEPTS_GUIDE.md, VIVA_QUESTIONS.md
```

## Running it locally

### 1. Train the model (produces `models/best_model.joblib`)

```bash
cd WindCast_AI
pip install -r api/requirements.txt
python src/eda.py        # optional: regenerate EDA figures
python src/train.py       # trains all 3 models, logs to MLflow, saves the best one
```

Inspect experiment runs with:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

### 2. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```
Interactive docs at `http://localhost:8000/docs`.

### 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
```
Opens on `http://localhost:5173`, proxying `/api/*` to the FastAPI server
on port 8000 (see `frontend/vite.config.ts`).

### 4. (Optional) Run the API in Docker

```bash
docker build -f docker/Dockerfile -t windcast-api .
docker run -p 8000:8000 windcast-api
```

## Model Results (from the included trained run)

| Model | Test MAE (kW) | Test RMSE (kW) | Test R² |
|---|---|---|---|
| Linear Regression | ~151 | ~203 | ~0.944 |
| Random Forest | ~6.1 | ~13.1 | ~0.9998 |
| **XGBoost (selected)** | **~7.2** | **~9.9** | **~0.9999** |

Re-running `src/train.py` regenerates these (values may shift slightly
with library versions / random seeds) and overwrites `models/best_model.joblib`.
