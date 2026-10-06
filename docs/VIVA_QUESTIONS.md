# WindCast AI — Viva & Interview Question Bank

Organized by topic. Each answer is written the length you'd actually say
out loud in a review — expand using `docs/CONCEPTS_GUIDE.md` if asked to
go deeper.

---

## Data & Preprocessing

**Q: Why did you interpolate missing values instead of dropping the rows?**
A: This is an hourly time series — dropping a row creates a gap that would
corrupt a chronological train/test split and skip an hour the forecasting
system is actually expected to cover. Since wind speed and temperature
change smoothly hour to hour, linear interpolation is a physically
reasonable way to fill small gaps; we still drop rows where the *target*
itself is missing, since we won't fabricate ground-truth power values.

**Q: Why clip outliers instead of removing them?**
A: A clipped row keeps its other, still-valid feature values instead of
losing the entire hour's data. We clip physically impossible values (e.g.
negative power) and use a 1.5×IQR clip on the target to tame extreme
sensor spikes without discarding real data.

**Q: Why engineer a `wind_speed_cubed` feature specifically?**
A: Turbine physics: power available in wind is proportional to velocity
cubed (`P ∝ v³`). Giving the model that relationship directly (instead of
hoping it infers a cubic curve from raw `wind_speed` alone) lets even
Linear Regression partially capture the non-linearity, and it ends up the
single most important feature for the tree models too.

**Q: Why extract hour/day/month instead of one-hot encoding them?**
A: Keeps the feature space small, and tree-based models (Random Forest,
XGBoost) can already recover cyclic/threshold patterns (e.g. "hour < 6")
from plain integers via repeated splits — one-hot encoding would add
dozens of sparse columns for no real benefit here.

---

## Time Series & Splitting

**Q: Why not use `train_test_split` with `shuffle=True`?**
A: Because adjacent hours are highly correlated (weather doesn't jump
randomly), a random split puts near-duplicate neighboring hours in both
train and test. The model then isn't tested on truly unseen conditions —
it's echoing a close neighbor — which **inflates** the reported accuracy
and is a textbook case of data leakage for time series.

**Q: Walk me through how TimeSeriesSplit works.**
A: It creates `n_splits` forward-chaining folds. Fold 1 trains on the
earliest chunk of data and validates on the next chunk; fold 2 trains on
the first two chunks and validates on the third chunk; and so on. Every
validation fold is always strictly *after* its own training data — so no
fold ever leaks future information backward — while still using multiple
folds for a more robust generalization estimate than one static split.

**Q: What's the practical difference between your final chronological
80/20 split and the TimeSeriesSplit cross-validation?**
A: The 80/20 split is the single hold-out set used for the final reported
test metrics (mimicking "deploy today, predict the future"). TimeSeriesSplit
is used *within* the training portion to get a more robust, multi-fold
estimate of generalization before ever touching the final test set —
similar in spirit to k-fold CV, but time-order-respecting.

---

## Models

**Q: Why train Linear Regression at all if it performs so much worse?**
A: It's the baseline. Without it, you can't tell how much value the more
complex models are actually adding — "R²=0.9999" only means something
relative to a simpler alternative. Here, the baseline error (~151 kW MAE)
being over 20x larger than XGBoost's (~7 kW) is itself strong evidence
that the underlying relationship is genuinely non-linear, which justifies
the added model complexity.

**Q: Explain bagging vs. boosting in one sentence each.**
A: Bagging (Random Forest) trains many independent trees on bootstrapped
samples in parallel and averages their predictions to reduce variance.
Boosting (XGBoost) trains trees sequentially, each one correcting the
residual errors of the ensemble so far, to reduce bias.

**Q: Why did XGBoost edge out Random Forest here?**
A: XGBoost's sequential error-correction combined with explicit
regularization (row/feature subsampling, L2 penalty on leaf weights) gave
it the lowest cross-validated MAE across TimeSeriesSplit folds and the
best RMSE/R² on the final hold-out — meaning it wasn't just lucky on one
test window, it was consistently strongest.

**Q: What does `max_depth` control, and what happens if you set it too
high?**
A: It caps how many splits deep a tree can grow. Too high, and a single
tree can carve out a leaf for nearly every individual training row —
memorizing noise instead of learning general patterns (overfitting); we
capped it at 14 (Random Forest) / 5 (XGBoost, deliberately shallow because
boosting builds strength through many weak trees, not one deep one).

**Q: What does `learning_rate` do in XGBoost, and why use a small value
like 0.05?**
A: It scales down each new tree's contribution to the ensemble. A smaller
learning rate means each round corrects errors more cautiously, requiring
more boosting rounds (`n_estimators`) to converge, but typically
generalizes better than taking large, aggressive correction steps.

**Q: How is feature importance computed for tree models?**
A: Every time a feature is used to split a node, the resulting reduction
in prediction error is recorded; importances are these reductions summed
across all trees and splits, then normalized. In our run, `wind_speed` and
`wind_speed_cubed` together account for ~99% of XGBoost's importance.

---

## Evaluation

**Q: Why report MAE, RMSE, AND R² instead of just one?**
A: They answer different questions. MAE gives the average error in kW,
directly interpretable by a non-technical stakeholder. RMSE also uses kW
but penalizes large misses more heavily — useful when big errors are
disproportionately costly (e.g. a forecast wildly wrong right before peak
demand). R² is scale-free and answers "how much of the variance did the
model explain overall" — good for quick comparison across models, but on
its own doesn't tell you the real-world error magnitude.

**Q: Your R² is 0.9999 — is something wrong?**
A: Not necessarily, for two reasons specific to this problem: (1) power
output for a well-instrumented turbine really is almost deterministically
a function of wind speed (physics, not noise), and (2) we specifically
avoided the random-split leakage that inflates R² artificially — this
number comes from the strict chronological hold-out set. It's worth
sanity-checking against MAE in real units (≈7 kW out of a ~1,000 kW
average) rather than trusting R² alone.

---

## MLOps

**Q: What's the difference between a normal ML workflow and MLOps?**
A: A normal workflow ends when a notebook prints a good accuracy number.
MLOps treats the model as a production software artifact: every training
run is reproducible and logged (params, metrics, artifacts), the winning
model is versioned in a registry (not just saved as a random file), it's
deployed behind a monitored API, and there's a defined plan for detecting
when it degrades and retraining it — the same rigor DevOps applies to
regular software, extended to cover data and models too.

**Q: What does MLflow actually give you here that a spreadsheet of
results wouldn't?**
A: Every run captures not just the final metric, but exactly which
hyperparameters and which model artifact produced it, in a queryable,
versioned store — you can reopen `windcast-power-forecaster` version 1 six
months from now and get back the *exact* fitted model object, not just a
number in a spreadsheet you'd have to somehow reproduce.

**Q: Explain data drift vs. model drift vs. concept drift.**
A: Data drift = the INPUT distribution shifts (e.g. average wind speed
changes seasonally) even though the underlying wind→power relationship is
unchanged. Model drift = live prediction quality degrades over time even
though inputs look normal (e.g. blade wear changes real output for the
same wind speed). Concept drift = the actual input→output relationship
itself changes (e.g. a firmware update alters the turbine's power curve).
Data drift is the easiest to detect early (you don't need new labels,
just input statistics); concept/model drift need fresh ground-truth data
to catch.

**Q: How would you decide when to retrain?**
A: Two complementary triggers: a schedule (e.g. weekly, as a baseline
safety net so the model keeps up with slow seasonal drift automatically),
and a trigger-based check (if the drift monitor flags a statistically
significant shift in input distribution, or live MAE — once ground truth
catches up — exceeds an acceptable threshold, kick off retraining
immediately rather than waiting for the schedule).

**Q: How does rollback work if a newly deployed model performs worse?**
A: Because the Model Registry keeps every version rather than overwriting,
rollback just means re-pointing the serving layer at the previous
registered version — no retraining needed, and it can be done in minutes.

---

## Deployment

**Q: Walk me through what happens when the frontend calls `/predict`.**
A: The React form sends a JSON POST with weather features → FastAPI's
Pydantic model validates types and physically sane ranges → the backend
computes `wind_speed_cubed` (identical to the training-time feature
engineering, imported from the same `src/data_pipeline.py` module, so
there's no train/serve mismatch) → assembles a DataFrame with columns in
the exact order the model was trained on → `model.predict()` → the
resulting float is wrapped in a typed JSON response and rendered on the
dashboard.

**Q: Why load the model once at startup instead of on every request?**
A: Deserializing a ~1MB joblib file involves disk I/O and CPU work;
doing that on every single API call would add unnecessary latency for no
benefit, since the model doesn't change between requests — it's only
updated when a new training run replaces the artifact.

**Q: Why containerize the API with Docker?**
A: To guarantee the exact same Python interpreter, library versions
(scikit-learn/XGBoost version mismatches can even fail to unpickle a
model), and code that passed testing is what actually runs in production
— eliminating "works on my machine" failures, and making horizontal
scaling as simple as running the same image multiple times behind a load
balancer.

**Q: How does the frontend avoid hardcoding `localhost:8000`?**
A: All calls go through a single typed API client (`src/lib/api.ts`) that
reads a base URL from an environment variable, defaulting to `/api`. In
development, Vite proxies `/api/*` to the local FastAPI server; in
production, the same relative path is proxied by Nginx or a load balancer
sitting in front of both services — so the exact same frontend build works
in both environments without code changes.

---

## "Why" Questions You're Likely to Get Rapid-Fire

- **Why XGBoost?** Best cross-validated + hold-out accuracy, built-in
  regularization, industry-standard for tabular/structured data.
- **Why Random Forest as a middle option?** Nearly as accurate, simpler to
  reason about (parallel bagged trees, no sequential error-correction to
  explain), a good robustness check against XGBoost.
- **Why MAE as the headline metric on the dashboard?** Most directly
  interpretable in the target's real unit (kW) for a non-technical
  stakeholder like a grid operator.
- **Why MLOps for a student project?** Demonstrates the difference between
  "a model that scored well once" and "a model I could actually hand to an
  operations team" — exactly the gap between an academic exercise and
  industry-ready work.
- **Why TimeSeriesSplit?** Prevents data leakage from adjacent correlated
  hours; gives an honest estimate of forecasting accuracy on genuinely
  future, unseen conditions.
- **How does deployment work end-to-end?** Trained model (`src/train.py`)
  → saved artifact + MLflow registry entry → loaded once by FastAPI
  (`api/main.py`) → containerized with Docker → served behind the React
  dashboard, which calls it over HTTP for both single predictions and
  power-curve simulations.
- **How does the API make predictions?** Validates the request, applies
  the same feature engineering as training, orders columns to match the
  training schema, and calls `model.predict()` — the exact same call
  pattern used during evaluation, just on one row at a time.
