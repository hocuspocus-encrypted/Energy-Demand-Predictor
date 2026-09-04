# Setup Guide

Local development, training, orchestration, and deployment instructions
for WattFlow. For what the project is and why it's
built this way, see [README.md](README.md).

## Setup

```bash
pip install -r requirements.txt
```

## Quick start (run in order)

```bash
python src/fetch_data.py --synthetic   # 1. generate/download data
python src/features.py                 # 2. build features
python src/train.py --run-name baseline_rf   # 3. train + log to MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db   # 4. (optional) view results
```

## Day 1–2 pipeline

### 1. Get data — `src/fetch_data.py`

Two modes:

**Real Kaggle data (what you'll actually use):**
```bash
pip install kaggle
# Kaggle account -> Settings -> API -> Create New Token, save as ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
python src/fetch_data.py --dataset robikscube/hourly-energy-consumption
```
Swap `--dataset` for any Kaggle slug with an hourly `datetime` + numeric
target column (weather datasets like `selfishgene/historical-hourly-weather-data`
work too — merge them on timestamp before running `features.py` if you use
two separate datasets).

**Synthetic fallback (no credentials needed, runs anywhere):**
```bash
python src/fetch_data.py --synthetic
```
This generates 2 years of hourly `datetime, temp_c, humidity, wind_speed,
energy_mw` with realistic seasonal, daily, weekday, temperature-driven,
and autocorrelated structure — same schema as the real merged dataset, so
every downstream script works unmodified once you switch to real data.

Output: `data/raw/energy_weather_raw.csv`

### 2. Build features — `src/features.py`

```bash
python src/features.py
```

Scikit-learn-based feature pipeline:
- Calendar features: hour, day-of-week, month, weekend flag
- Cyclical encodings (sin/cos of hour-of-day and day-of-year, so hour 23
  and hour 0 aren't seen as far apart)
- Lag features: 1h, 2h, 3h, 24h (yesterday same hour), 168h (last week same hour)
- Rolling stats: 24h and 168h rolling mean/std of energy, rolling mean of temp
- `build_preprocessing_pipeline()` returns a picklable `sklearn.Pipeline`
  (StandardScaler via ColumnTransformer) reused identically in training and
  inference — the core reproducibility guarantee for this stage.

Output: `data/processed/features.csv`

### 3. Train + track — `src/train.py`

```bash
python src/train.py --run-name baseline_rf
```

- Chronological train/test split (never shuffle time-series data)
- RandomForestRegressor wrapped in the same preprocessing pipeline
- Logs to a local MLflow tracking store (`sqlite:///mlflow.db`):
  - **Params:** model type, n_estimators, max_depth, min_samples_leaf,
    feature count, row counts, random_state
  - **Metrics:** train/test MAE, RMSE, MAPE
  - **Artifacts:** the fitted pipeline (preprocessing + model, loadable
    with `mlflow.sklearn.load_model`), plus the exact feature column list
    used, for reproducibility

Try different hyperparameters and compare runs:
```bash
python src/train.py --run-name rf_deep --n-estimators 500 --max-depth 20
python src/train.py --run-name rf_shallow --max-depth 6
```

View the tracking UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000
```

## Days 3–4: validation, retraining, orchestration

### 4. Validate data — `scripts/validate_data.py`

```bash
python scripts/validate_data.py
```

Runs against the raw fetched CSV, before features are built. Splits checks
into two tiers:
- **Blocking** (exits 1, stops the pipeline): missing/wrong-typed columns,
  duplicate timestamps, nulls in the target column (`energy_mw`)
- **Warning** (logged, doesn't stop the pipeline): out-of-range
  `temp_c`/`humidity`/`wind_speed`/`energy_mw` values, gaps in the hourly
  time series

### 5. Retrain with champion/challenger — `scripts/retrain_pipeline.py`

```bash
python scripts/retrain_pipeline.py --run-name rf_challenger --n-estimators 500
```

Wraps `run_training()` (the same training logic `train.py`'s CLI uses,
`src/train.py` refactored so it's callable, not just invocable from the
command line). Registers the new run as a version of the
`wattflow` MLflow registered model, then promotes it to the
`@champion` alias only if its test MAE beats the current champion's (or if
there is no champion yet). Writes the outcome to `data/retrain_status.json`.

### 6. Orchestrate — `dags/energy_demand_pipeline.py`

An Airflow 3.x TaskFlow DAG (`airflow.sdk`, not the older
`airflow.decorators` path) that runs the whole thing on a simulated daily
schedule:

```
fetch_data → validate_data → build_features → retrain_model → report_outcome
```

Each task shells out to the scripts above, so the DAG is a thin scheduling
layer over code that's also runnable by hand.

Run it via Docker:

```bash
cd docker
docker compose up -d --build
# UI at http://localhost:8080
docker compose exec airflow-scheduler airflow dags trigger energy_demand_pipeline
```

`docker/Dockerfile` extends `apache/airflow:3.3.1` with this project's
`requirements.txt`. `docker/docker-compose.yaml` runs a `LocalExecutor`
stack (postgres, api-server, scheduler, dag-processor, triggerer — no
Celery/Redis/worker, which is overkill for a single machine) and
bind-mounts the whole repo into each container at `/opt/airflow/project`.

Auth is Airflow 3's `SimpleAuthManager` rather than the default FAB auth
manager — the FAB provider's `airflow users create` / `airflow roles list`
CLI commands are broken against `apache/airflow:3.3.1`
(`AirflowSecurityManagerV2` is missing `find_role`/`get_all_roles`), so
`AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS=true` sidesteps that
entirely and treats every authenticated request as admin — fine for a
single-machine dev setup, not for anything exposed beyond localhost.

Verified end-to-end: built the image, brought the stack up, triggered the
DAG via CLI, and confirmed all 5 tasks (`fetch_data` → `validate_data` →
`build_features` → `retrain_model` → `report_outcome`) succeeded, with
`retrain_status.json` reflecting a real champion/challenger comparison
against the model trained in the manual runs.

> **Watch out:** the `fetch_data` task always runs with `--synthetic`
> (containers don't have Kaggle credentials), and its containers have
> `restart: unless-stopped`. Left running with the `@daily` schedule, it
> will silently retrain on synthetic data and can promote a
> synthetic-scale champion over a real-data one, since the
> champion/challenger gate compares raw MAE and can't tell the two
> datasets' scales apart. `docker compose down` (not just stopping the
> containers) before walking away avoids it restarting itself the next
> time Docker starts. If you bring the stack back up, re-run
> `scripts/export_champion.py` afterward and check
> `data/retrain_status.json`/the `@champion` alias didn't flip to a run
> trained on the wrong data.

## Day 5: Serve the model — `src/serve.py`

```bash
python scripts/export_champion.py   # once, or whenever the champion changes
uvicorn src.serve:app --reload
# docs at http://127.0.0.1:8000/docs
```

`scripts/export_champion.py` resolves the `wattflow`
registry's `@champion` alias (the version `scripts/retrain_pipeline.py`
promotes) and exports it to a standalone bundle at `model/` — a joblib
pipeline, `feature_columns.json`, and `metadata.json` (run id, version,
test metrics). It's loaded via `runs:/<run_id>/model` rather than
`models:/wattflow@champion` directly, since the alias-URI
form hits a Windows-specific bug in mlflow's artifact resolution where a
local temp path gets misparsed as a URI with scheme `c` (from the drive
letter).

`src/serve.py` then loads *only* from that `model/` bundle — no mlflow,
mlflow.db, or mlruns/ needed at serve time. That keeps the deployed
service small (the bundle is tens of MB; the full tracking store is
hundreds) and reproducible: `requirements-serve.txt` lists just
FastAPI/scikit-learn/pandas/joblib, not the training-side dependencies in
`requirements.txt`. Endpoints:

- `GET /health` — liveness + which model version is loaded
- `GET /sample` — a random real historical row (its 22 engineered
  features plus the true `energy_mw` that occurred), so a client can
  prefill a prediction form instead of hand-typing engineered features
- `POST /predict` — takes the same 22 features `feature_columns.json`
  lists (the logged pipeline only does scaling, not lag/rolling
  recomputation) and returns `{predicted_energy_mw, model_version, run_id}`

The response schema is validated at load time against the bundle's
`feature_columns.json`, so a mismatched model/API version fails fast at
boot instead of mispredicting silently.

```bash
curl http://127.0.0.1:8000/sample
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{...}'
# {"predicted_energy_mw":31519.5,"model_version":"13","run_id":"5237e7c89a3441aabe69d0f4d7430f61"}
```

### Deploying: FastAPI backend on Render, UI on Vercel

**Backend (Render)** — `Dockerfile.serve` builds a serving-only image from
`requirements-serve.txt` and the `model/` bundle (committed to the repo,
unlike `mlruns/`/`mlflow.db`). `render.yaml` declares it as a Docker web
service with a `/health` check. Connect the repo on
[render.com](https://render.com), it picks up `render.yaml` automatically;
after the frontend is deployed, set the `ALLOWED_ORIGINS` env var on the
Render service to its Vercel URL (comma-separated if there's more than
one, e.g. a preview + production domain) so CORS isn't wide open in prod.

**Frontend (Vercel)** — `frontend/` is a Next.js app (`app/page.tsx`) that
calls the backend via `NEXT_PUBLIC_API_URL`. Import the repo on
[vercel.com](https://vercel.com) with **Root Directory** set to
`frontend`, and set `NEXT_PUBLIC_API_URL` to the Render service's URL in
the project's environment variables. Locally, copy
`frontend/.env.local.example` to `frontend/.env.local` and point it at
wherever `uvicorn src.serve:app` is running.

```bash
cd frontend
npm install
npm run dev   # http://localhost:3000, calling NEXT_PUBLIC_API_URL
```

## Day 6-7: Containerize the full pipeline — `Dockerfile`

```bash
docker build -t wattflow-pipeline .
docker run --rm -v "$(pwd):/app" wattflow-pipeline   # synthetic data, default
```

Runs `fetch_data → validate_data → build_features → retrain_pipeline` end
to end in one container via `scripts/run_pipeline.py` — the same steps the
Airflow DAG chains, callable as a single command without needing Airflow
running at all. This is a third, distinct Dockerfile in the repo:

| File | Purpose |
|---|---|
| `Dockerfile` | one-shot full pipeline run (this section) |
| `Dockerfile.serve` | serving-only image, deployed to Render |
| `docker/Dockerfile` | Airflow orchestration image, runs the DAG on a schedule |

Mount the repo (`-v "$(pwd):/app"`) so `mlflow.db`, `mlruns/`, and `data/`
persist on the host instead of vanishing with the container. Without a
Kaggle dataset specified it defaults to `--synthetic`, so it runs with zero
credentials out of the box. To train on real data, pass Kaggle credentials
through and override the args (anything after the image name goes to
`scripts/run_pipeline.py`, replacing the Dockerfile's default `CMD`):

```bash
docker run --rm -v "$(pwd):/app" \
  -e KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)" \
  wattflow-pipeline \
  --dataset robikscube/hourly-energy-consumption --kaggle-file PJME_hourly.csv \
  --n-estimators 500 --max-depth 20
```

After a run, `scripts/export_champion.py` (outside the container, or
appended as a second `docker run`) picks up the new champion for serving
if it got promoted.

## Project layout

```
energy-demand-mlops/
├── data/
│   ├── raw/            # fetch_data.py output
│   ├── processed/       # features.py output
│   └── retrain_status.json   # retrain_pipeline.py's last champion/challenger outcome
├── src/
│   ├── fetch_data.py    # Day 1: data acquisition (Kaggle + synthetic fallback)
│   ├── features.py      # Day 1-2: sklearn feature engineering
│   ├── train.py         # Day 2: training + MLflow tracking (run_training() is importable)
│   └── serve.py         # Day 5: FastAPI /predict endpoint over the champion model
├── scripts/
│   ├── validate_data.py     # Day 3: blocking + warning data checks
│   ├── retrain_pipeline.py  # Day 3-4: champion/challenger retraining
│   ├── export_champion.py   # Day 5: export @champion to model/ for serving
│   └── run_pipeline.py      # Day 6-7: fetch->validate->features->retrain in one call
├── dags/
│   └── energy_demand_pipeline.py   # Day 4: Airflow TaskFlow DAG
├── docker/
│   ├── Dockerfile            # apache/airflow:3.3.1 + requirements.txt
│   └── docker-compose.yaml   # LocalExecutor Airflow stack
├── model/                # Day 5: standalone champion bundle (committed, unlike mlruns/)
│   ├── pipeline.joblib
│   ├── feature_columns.json
│   ├── metadata.json
│   └── samples.json      # real rows for the UI's "load real example"
├── frontend/              # Day 5: Next.js UI, deployed to Vercel
├── Dockerfile             # Day 6-7: full-pipeline image (see above)
├── Dockerfile.serve       # Day 5: serving-only image, deployed to Render
├── render.yaml            # Day 5: Render service definition
├── mlflow.db              # MLflow SQLite tracking store (created on first run)
├── requirements.txt       # full deps: training + serving
└── requirements-serve.txt # Day 5: serving-only deps (no mlflow/kaggle/airflow)
```

## Notes for later in the week

- `build_preprocessing_pipeline` and the RandomForest are already combined
  into a single `sklearn.Pipeline`, logged as one MLflow artifact — this is
  what you'll load back for batch/online inference later in the week.
- `feature_columns.json` is logged with every run specifically so a future
  serving script can validate its input schema against the exact features
  a given model version was trained on.
- Swap in real Kaggle data any time — nothing downstream changes, since
  `fetch_data.py` writes to the same `energy_weather_raw.csv` schema either way.
