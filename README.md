# Energy Demand Predictor — Week 2, Days 1–4

Model lifecycle automation, reproducibility, and experiment tracking
for an hourly energy-demand forecasting model.

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
`energy-demand-forecaster` MLflow registered model, then promotes it to the
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
│   └── train.py         # Day 2: training + MLflow tracking (run_training() is importable)
├── scripts/
│   ├── validate_data.py     # Day 3: blocking + warning data checks
│   └── retrain_pipeline.py  # Day 3-4: champion/challenger retraining
├── dags/
│   └── energy_demand_pipeline.py   # Day 4: Airflow TaskFlow DAG
├── docker/
│   ├── Dockerfile            # apache/airflow:3.3.1 + requirements.txt
│   └── docker-compose.yaml   # LocalExecutor Airflow stack
├── mlflow.db             # MLflow SQLite tracking store (created on first run)
└── requirements.txt
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
