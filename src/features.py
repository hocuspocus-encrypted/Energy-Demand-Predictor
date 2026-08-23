"""
Day 1-2: Feature engineering with scikit-learn.

Builds a clean, reusable sklearn Pipeline that turns the raw
datetime/temp/humidity/wind/energy table into a model-ready feature
matrix: calendar features, cyclical encodings, lag features, and
rolling weather stats.

The transformers are stateless functions wrapped in FunctionTransformer
plus a ColumnTransformer, so the exact same pipeline object can be
pickled and reused at inference time (important for reproducibility --
this is the piece MLflow will version alongside the model).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

RAW_FILE = Path(__file__).resolve().parents[1] / "data" / "raw" / "energy_weather_raw.csv"
PROCESSED_FILE = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv"

TARGET = "energy_mw"

LAG_HOURS = [1, 2, 3, 24, 168]        # 1-3h, same-hour-yesterday, same-hour-last-week
ROLLING_WINDOWS = [24, 168]           # 1-day and 1-week rolling stats


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["datetime"]
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # cyclical encodings so the model sees hour 23 and hour 0 as close
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lag in LAG_HOURS:
        df[f"energy_lag_{lag}h"] = df[TARGET].shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"energy_roll_mean_{window}h"] = (
            df[TARGET].shift(1).rolling(window).mean()
        )
        df[f"energy_roll_std_{window}h"] = (
            df[TARGET].shift(1).rolling(window).std()
        )
        df[f"temp_roll_mean_{window}h"] = (
            df["temp_c"].shift(1).rolling(window).mean()
        )
    return df


def build_feature_frame(raw_path: Path = RAW_FILE) -> pd.DataFrame:
    df = pd.read_csv(raw_path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    df = df.dropna().reset_index(drop=True)  # drop rows with incomplete lag/rolling windows
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"datetime", TARGET}
    return [c for c in df.columns if c not in exclude]


def build_preprocessing_pipeline(numeric_features: list[str]) -> Pipeline:
    """Reusable, picklable preprocessing pipeline (scaling only -- all
    features here are already numeric after build_feature_frame)."""
    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), numeric_features)],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor)])


def main():
    df = build_feature_frame()
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    feat_cols = get_feature_columns(df)
    print(f"Built {len(df):,} rows x {len(feat_cols)} features -> {PROCESSED_FILE}")
    print("Feature columns:", feat_cols)


if __name__ == "__main__":
    main()
