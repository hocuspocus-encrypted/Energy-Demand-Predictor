# Energy Demand Predictor — Week 2, Days 1–2

Model lifecycle automation, reproducibility, and experiment tracking
for an hourly energy-demand forecasting model.

## Setup

```bash
pip install -r requirements.txt
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

## Project layout

```
energy-demand-mlops/
├── data/
│   ├── raw/            # fetch_data.py output
│   └── processed/       # features.py output
├── src/
│   ├── fetch_data.py    # Day 1: data acquisition (Kaggle + synthetic fallback)
│   ├── features.py      # Day 1-2: sklearn feature engineering
│   └── train.py         # Day 2: training + MLflow tracking
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
