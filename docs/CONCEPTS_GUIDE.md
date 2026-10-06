# WindCast AI — Concepts Guide (Mentor Notes)

This is the "why", taught the way a senior engineer would explain it to a
junior on the team. Every section maps to real files in this repo, so you
can point at working code while you explain the idea.

---

## 1. The Data Pipeline (`src/data_pipeline.py`)

**Analogy first:** think of the raw CSV like a turbine's logbook kept by a
tired night-shift operator — mostly accurate, but with the occasional
duplicate entry, a blank cell where a sensor glitched, and a wildly wrong
number scribbled during a storm. You wouldn't discard the whole logbook;
you'd fix the obvious errors and use the rest.

**Dataset structure.** One row per hour, for one year (8,760 rows), six
columns: timestamp, power generated (target), wind speed, wind direction,
pressure, air temperature.

**Why each feature affects power generation:**
- `wind_speed` — the dominant driver. A turbine converts kinetic energy in
  moving air into electricity; more speed, more available energy.
- `wind_direction` — matters less for OUR model because a well-controlled
  turbine yaws to face the wind, so direction mostly affects *efficiency*
  at the margins rather than raw availability.
- `pressure` — a proxy for air density (denser air carries more kinetic
  energy at the same speed).
- `air_temperature` — also a density proxy (hot air is less dense).

**Cleaning strategy (and why each step exists — see the docstring in
`clean()`):**
1. *Duplicate removal* — a repeated timestamp lets the model "memorize" a
   row instead of generalizing.
2. *Timestamp parsing* — required to derive hour/day/month and to sort
   chronologically for the time-series split later.
3. *Missing values* — we do NOT just drop rows in an hourly series, because
   that creates a "hole" that corrupts a chronological split. We
   interpolate small predictor gaps (wind/temperature change smoothly
   hour-to-hour) and only drop rows where the *target* is missing — we
   won't invent ground truth.
4. *Outlier handling* — physically impossible values (negative power,
   pressure outside ~0.9–1.1 atm) are **clipped, not deleted**, so we keep
   the rest of that row's information; an IQR-based clip tames a handful
   of extreme sensor spikes in the target.

**Common mistake to avoid:** dropping every row with any NaN. On an hourly
time series that silently deletes real chronological structure and can
bias a time-based split.

**Feature engineering — why each new feature exists:**
- `hour`, `day`, `month` — let the model learn diurnal/seasonal wind
  patterns without one-hot exploding the feature space; tree models split
  on numeric thresholds naturally (e.g. "hour < 6").
- `wind_speed_cubed` — turbine physics: power in wind ∝ `v³`
  (`P = 0.5 × ρ × A × v³`, before the turbine's own efficiency curve and
  cut-in/cut-out limits apply). Feeding `v³` directly lets even a
  **linear** model capture this cubic relationship, and it is consistently
  the #1 feature by importance for the tree models too (see Model
  Performance page → 63% importance for XGBoost in our run).

---

## 2. Exploratory Data Analysis (`src/eda.py`, `outputs/figures/`)

| Plot | Why it's used | Review talking point |
|---|---|---|
| Histograms | Shows the *shape* of each variable's distribution | Power is right-skewed with a spike near 0 kW — calm hours |
| Boxplots | Flags spread and outliers on a common scale | Justifies the IQR clip in cleaning |
| Correlation heatmap | Linear relationship strength between every pair | wind_speed / wind_speed³ correlate ~0.95 with power — previews feature importance |
| Scatterplot (wind vs power) | Reveals the *actual functional shape*, not just linear strength | Shows the textbook S-shaped turbine power curve — near-zero below cut-in, steep rise, flat plateau near rated power |

**Why the scatterplot matters more than it looks:** the correlation
heatmap only tells you the LINEAR strength, but the true relationship is
S-shaped (sigmoid-like). That single observation is the entire justification
for (a) engineering `wind_speed_cubed`, and (b) expecting tree ensembles to
outperform plain Linear Regression.

---

## 3. Machine Learning Models (`src/train.py`)

### 3.1 Linear Regression — the baseline

**Intuition:** fit the straight line (or hyperplane, with multiple
features) that minimizes the sum of squared errors between predictions and
actual power output.

**Math:** `ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ`, weights chosen by minimizing
`Σ(y - ŷ)²` (ordinary least squares).

**Advantages:** fast, interpretable (each weight is a direct effect size),
no hyperparameters to tune, a great sanity-check baseline.

**Disadvantages:** assumes the relationship is linear. Our data is NOT
linear (see the power curve) — hence the ~151 kW MAE baseline vs ~6-7 kW
for the ensembles.

**Why it's still worth training:** a baseline tells you *how much* the
fancier models are actually buying you. Without it, "R²=0.9999" sounds
impressive but you can't tell if a much simpler model gets you 90% of the
way there for free. Here it clearly does not — the baseline error is over
20x larger.

### 3.2 Random Forest Regressor

**Building blocks:**
- *Decision Tree:* recursively splits the data on the feature/threshold
  that most reduces prediction error (e.g. "is wind_speed < 6.2?"), down to
  leaves that output an average value.
- *Bagging (Bootstrap Aggregating):* train many trees, each on a random
  resampled ("bootstrapped") subset of the training rows, then average
  their predictions. This cancels out each tree's individual overfitting.
- *Ensemble Learning:* the general principle that combining many "weak-ish"
  models produces a stronger, more stable model than any single one.
- *Feature Importance:* for each tree, track how much each feature
  reduced error every time it was used to split, and average across all
  trees — gives a ranked list of which inputs matter most.

**Why it performs well on tabular data:** tabular features like ours
(wind speed, direction, pressure...) rarely have a clean linear
relationship, but trees naturally carve up feature space into regions with
different average outputs — no need to hand-engineer every non-linearity.

**Hyperparameters used here** (see `train.py`):
- `n_estimators=300` — number of trees; more trees lower variance with
  diminishing returns (300 is comfortably past that knee for 8.7k rows).
- `max_depth=14` — caps how deep each tree can grow, limiting overfitting.
- `min_samples_leaf=3` — a leaf must cover ≥3 samples, which smooths
  predictions instead of memorizing single rows.

### 3.3 XGBoost Regressor (our selected model)

**Building blocks:**
- *Gradient Boosting:* instead of averaging independent trees (bagging),
  build trees **sequentially**, where each new tree is trained to predict
  the *residual errors* of the trees before it.
- *Sequential Learning / Error Correction:* tree 1 makes a rough guess;
  tree 2 learns to correct tree 1's biggest mistakes; tree 3 corrects
  what's left over, and so on — the ensemble total is the sum of all
  trees' outputs.
- *Regularization:* XGBoost adds explicit penalty terms on tree complexity
  (leaf weights, tree depth) directly into the loss function it optimizes,
  which is why it tends to generalize better than a plain gradient-boosted
  tree implementation.

**Why XGBoost is common in industry:** it wins the accuracy/speed
trade-off on structured/tabular data extremely consistently (a huge share
of Kaggle competition winners on tabular problems used it), has built-in
handling for missing values, and its regularization makes it less prone to
overfitting out of the box than a vanilla boosting implementation.

**Hyperparameters used here:**
- `n_estimators=400` — boosting rounds (sequential trees).
- `learning_rate=0.05` — shrinks each tree's contribution; small values
  need more rounds but generalize better ("learn slowly, correct often").
- `max_depth=5` — each round's tree is shallow (a "weak learner" by
  design — the *ensemble*, not any one tree, should be strong).
- `subsample=0.8` — each round trains on a random 80% of rows
  (regularization via row sampling).
- `colsample_bytree=0.8` — each round only considers 80% of features
  (regularization via feature sampling, also decorrelates trees).
- `reg_lambda=1.0` — L2 penalty on leaf weights, discourages any single
  leaf from making an extreme prediction.

**What happens internally during training (all 3 models):**
`model.fit(X_train, y_train)` — Linear Regression solves the normal
equations (or gradient descent) once; Random Forest grows N independent
trees in parallel on bootstrapped samples; XGBoost grows N trees one after
another, each fit to the previous ensemble's residuals, each round scaled
by `learning_rate`.

**How predictions are generated:**
`model.predict(X_test)` — Linear Regression: dot product of learned
weights and input features. Random Forest: average of all trees' leaf
outputs. XGBoost: sum of all trees' leaf outputs (scaled by learning rate).

---

## 4. Time Series Considerations (`chronological_split`, `time_series_cv_score` in `train.py`)

**Why random `train_test_split` causes data leakage here:** wind speed at
2:00am and 3:00am on the same night are highly correlated (weather doesn't
jump randomly hour to hour). A random 80/20 split puts some of those
neighboring hours in train and others in test — the model isn't being
tested on truly unseen conditions, it's echoing a near-duplicate
neighbour. This **inflates** the reported R²/MAE and doesn't reflect how
the model will really be used (forecasting the *future* from the *past*).

**Why chronological split is better:** train on the earlier months, test
on the later, untouched months — this mirrors production use exactly.

**How `TimeSeriesSplit` works:** instead of one arbitrary cut, it makes
`n_splits` forward-chaining folds. Fold 1 trains on the earliest chunk and
validates on the next chunk; fold 2 trains on the first two chunks and
validates on the third; and so on. Every validation fold is always
strictly *after* its training data — no fold ever leaks future information
backward — giving a more robust generalization estimate than a single
split, while still respecting time order.

**How to justify it in a review:** "We used a chronological 80/20 split
plus 5-fold TimeSeriesSplit cross-validation instead of random
train_test_split, because a random split on hourly weather data leaks
information between adjacent, highly-correlated hours and overstates real
-world accuracy."

---

## 5. Model Evaluation

| Metric | Formula | Intuition | Advantage | Limitation |
|---|---|---|---|---|
| **MAE** | `mean(\|y - ŷ\|)` | Average error in the *original units* (kW) | Easy to explain to non-technical stakeholders; robust to outliers | Treats all errors equally — doesn't penalize big misses extra |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Also in kW, but squares errors first | Penalizes large errors more — useful when big misses are more costly | Harder to interpret intuitively; sensitive to outliers |
| **R²** | `1 - SS_res/SS_tot` | Fraction of variance in power explained by the model | Scale-free, easy single-number summary | Can look artificially high with leakage (see Section 4); doesn't tell you the error's real-world magnitude |

**How to interpret our actual results:** XGBoost's Test MAE ≈ 7.2 kW on a
target whose mean is ~1,000 kW — average error is under 1% of typical
output. R² = 0.9999 means the model explains essentially all the variance
in power output from weather alone, which makes physical sense given how
strongly power is a deterministic function of wind speed for a real
turbine's power curve.

---

## 6. Feature Importance

**How it's calculated (tree models):** every time a feature is used to
split a node, the resulting reduction in error (impurity) is recorded;
importances are these reductions summed across every tree and split,
normalized to sum to 1 (or 100%).

**Why wind_speed (and its cubed feature) dominate:** it's the direct
physical driver of power output (`P ∝ v³`); everything else (direction,
pressure, temperature, time-of-day) only nudges efficiency at the margins.

**How to present it in a review:** show the bar chart (Model Performance
page), lead with the physics ("power scales with the cube of wind speed"),
then point out the model *discovered this from data* — it wasn't told the
physics, it learned it, which is strong evidence the model generalizes for
the right reasons rather than overfitting noise.

---

## 7. Model Selection

| Model | Test MAE (kW) | Test RMSE (kW) | Test R² |
|---|---|---|---|
| Linear Regression | ~151 | ~203 | ~0.944 |
| Random Forest | ~6.1 | ~13.1 | ~0.9998 |
| **XGBoost (selected)** | **~7.2** | **~9.9** | **~0.9999** |

*(exact numbers are written to `outputs/reports/model_comparison.csv` and
`models/model_metadata.json` on every training run — these will vary
slightly with library versions/random seeds.)*

**Why XGBoost was selected:** best RMSE and R² on the untouched
chronological hold-out set, and the lowest cross-validated MAE across all
5 TimeSeriesSplit folds — meaning it's not just lucky on one test window,
it's consistently the strongest across different time periods.

**Review-ready explanation:** "We compared three models on an identical
chronological hold-out set. Linear Regression set an interpretable
baseline but underfits the non-linear turbine power curve. Both ensembles
captured that non-linearity, with XGBoost edging out Random Forest on
RMSE and cross-validated stability, so it was registered as the production
model."

---

## 8. MLOps — Taught Like a Mentor

**What MLOps is:** the discipline of treating a trained model like a
*production software artifact* — versioned, tested, monitored, and
re-deployable — rather than a one-off notebook output. Borrows heavily
from DevOps (CI/CD, infrastructure-as-code) but adds ML-specific concerns:
data versioning, experiment tracking, model drift.

**Why companies use it:** a notebook that produced 94% accuracy once,
six months ago, on a laptop nobody can find, is not a product. Companies
need to know *exactly* which data + code + hyperparameters produced the
model currently serving traffic, be able to roll it back in minutes if it
breaks, and detect automatically when it starts degrading.

**MLOps vs. a traditional ML workflow:**
| Traditional | MLOps |
|---|---|
| Train once in a notebook | Training is a reproducible, re-runnable pipeline (`src/train.py`) |
| Best model = whichever cell you last ran | Every run is logged with params/metrics (MLflow); best is chosen systematically |
| Model file emailed around / saved ad hoc | Model Registry with versions or a joblib artifact + metadata file |
| No idea if it's still accurate 3 months later | Monitoring + drift detection + scheduled retraining |

### MLflow Integration (`src/train.py`, tracked to `mlflow/mlflow.db`)

- **Experiment Tracking** — every training run (one per model) is grouped
  under the `windcast-ai-power-forecasting` experiment, so all runs are
  comparable side by side.
- **Parameter Logging** (`mlflow.log_params`) — records exact
  hyperparameters (e.g. `max_depth=5`) so a result can always be
  reproduced.
- **Metric Logging** (`mlflow.log_metrics`) — records CV MAE, test MAE/
  RMSE/R² per run, so runs can be compared without re-running anything.
- **Artifact Logging** (`mlflow.sklearn.log_model`) — saves the actual
  fitted model object alongside its metadata.
- **Model Registry** — the winning run is registered under the name
  `windcast-power-forecaster`; each registration creates a new *version*,
  so history is never overwritten.

**Why each feature exists / what problem it solves:** without tracking, "we
improved MAE by 30%" is an unverifiable claim after the fact. With MLflow,
every claim is backed by a run ID you can open and inspect months later.

### Model Versioning

**Why it's needed:** models silently go stale as real-world conditions
drift (new turbine hardware, a new site, seasonal shifts not seen in
training data). Without versioning you can't safely compare "the new model"
against "what's currently live," and you can't undo a bad deployment.

**Production vs. Staging:** a newly trained model is typically registered
as "Staging" first, validated against a shadow/holdout traffic sample or a
canary rollout, and only promoted to "Production" once it's confirmed to
be at least as good as the current production version.

**Rollback strategy:** because every version is preserved in the registry
(not overwritten), rolling back is just re-pointing the serving layer
(the FastAPI service's `MODEL_PATH`/registry reference) at the previous
version — no retraining required, and it can be done in minutes.

### Monitoring — Drift

- **Data drift** — the distribution of INPUT features changes over time
  (e.g. average wind speed shifts because of a seasonal change or a new
  sensor). Detected by comparing live input statistics (mean/std, or a
  statistical test like Kolmogorov–Smirnov / Population Stability Index)
  against the training distribution. **This project implements a simple
  version of this on the MLOps Monitoring dashboard page** — it compares
  the mean of each incoming feature during the session against the
  training set's mean/std as a z-score.
- **Model drift** — the model's live *prediction quality* degrades over
  time, even if inputs look normal (e.g. turbine blades wear down and the
  same wind speed now yields less power). Detected by comparing live
  predictions against ground truth once it becomes available.
- **Concept drift** — the actual relationship between inputs and target
  changes (e.g. a firmware update changes the turbine's power curve
  itself). This is the hardest to catch quickly, because it requires
  fresh labeled data to notice the *relationship* changed, not just the
  inputs.

**How companies detect them in practice:** dashboards tracking rolling
error metrics, automated statistical tests on incoming feature
distributions, and alerting thresholds that page an on-call engineer or
trigger automatic retraining.

### Retraining Strategy

- **Scheduled retraining** — re-run `train.py` on a fixed cadence (e.g.
  weekly), so the model keeps up with slow seasonal drift automatically.
- **Trigger-based retraining** — kick off retraining automatically when
  drift crosses a threshold, or when live MAE exceeds an acceptable bound.
- **New data ingestion** — append freshly logged SCADA rows to `data/`,
  re-run the same `data_pipeline.py` cleaning logic (single source of
  truth, see Section 1), and retrain; the chronological split naturally
  extends to include the new period.

**Industry best practice:** combine both — scheduled retraining as a
baseline safety net, plus trigger-based retraining for faster response to
sudden drift (e.g. a hardware fault).

---

## 9. Deployment

### FastAPI Backend (`api/main.py`)

**Request flow:** client POSTs JSON to `/predict` → Pydantic (`PredictRequest`)
validates types and physical ranges (e.g. `wind_speed` between 0-40 m/s) →
the SAME feature engineering used in training (`wind_speed_cubed`) is
applied → a DataFrame is built with columns in the EXACT order the model
was trained on (`FEATURE_COLUMNS`, imported from `src/data_pipeline.py` —
this is what prevents "training/serving skew") → `model.predict()` →
response.

**Response flow:** the predicted float is wrapped in a `PredictResponse`
Pydantic model, which FastAPI serializes to JSON automatically; CORS
middleware allows the React frontend (a different origin in dev) to call
the API directly from the browser.

**Model loading:** the model is loaded ONCE at import time (module-level,
not inside the endpoint function) — reloading a ~1MB joblib file on every
request would add unnecessary latency and disk I/O to every single call.

### Docker (`docker/Dockerfile`, `api/requirements.txt`)

**Containerization, in one sentence:** package the exact Python
interpreter, every dependency version, the trained model binary, and the
code into one immutable image, so "works on my machine" becomes "works
everywhere this image runs."

**Why it matters here specifically:** a stale scikit-learn/XGBoost version
can even fail to *unpickle* a model trained on a newer version — Docker
eliminates that entire class of bug.

**Benefits:** reproducible builds, trivial horizontal scaling (`docker run`
the same image N times behind a load balancer), and a clean separation
between "what code needs" (image) and "where it runs" (any container
host — a laptop, a VM, Kubernetes).

**Deployment workflow:** `docker build` the image → push to a registry →
the target environment pulls and runs it, typically behind a reverse
proxy/load balancer that also serves the built frontend.

### Frontend (`frontend/`)

A React + TypeScript app (Vite) styled with the same design tokens as the
approved Stitch mockup, with five pages: Dashboard, Forecasting, Analytics,
Model Performance, MLOps Monitoring. It talks to the FastAPI backend
entirely through `src/lib/api.ts` — a small typed `fetch` wrapper — so
every network call's shape matches the backend's Pydantic models in
exactly one place. In development, Vite proxies `/api/*` to
`localhost:8000` (see `vite.config.ts`); in production this proxy role is
typically played by Nginx or a cloud load balancer sitting in front of
both the built static frontend and the FastAPI container.

---

## 10. Project Structure — Why Each Folder Exists

```
WindCast_AI/
├── data/          Raw + versioned input data (TexasTurbine.csv)
├── notebooks/      Exploratory / scratch work — never the source of truth for production logic
├── models/          Serialized trained model + metadata JSON consumed by the API
├── src/             Reusable, importable pipeline code (data cleaning, training) — the single
                     source of truth shared by training AND the API, preventing train/serve skew
├── api/             FastAPI service, its own requirements.txt
├── frontend/        React + TypeScript dashboard (Vite)
├── mlflow/           MLflow's sqlite tracking store (experiment/run history)
├── docker/           Containerization config
├── outputs/          Generated artifacts: EDA figures, model comparison reports
└── docs/             This guide + viva questions
```

**Why `src/` is separate from `api/`:** the API should be a thin
consumer of the pipeline, not where cleaning/feature-engineering logic
lives. If those steps were duplicated inside `api/main.py`, a future
change to cleaning logic could easily be updated in one place and
forgotten in the other — silently breaking predictions. Importing
`FEATURE_COLUMNS` and the cleaning functions from `src/data_pipeline.py`
in both `train.py` and `api/main.py` guarantees they can never drift apart.
