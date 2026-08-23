"""
Day 1-2: Data acquisition.

Primary path: pull an hourly energy-consumption dataset from Kaggle
(e.g. robikscube/hourly-energy-consumption, which is PJM grid data)
plus an hourly weather dataset, and merge them on timestamp.

Fallback path: if Kaggle credentials aren't configured (no
~/.kaggle/kaggle.json / KAGGLE_USERNAME+KAGGLE_KEY env vars), or the
`kaggle` package isn't installed, generate a synthetic dataset with an
identical schema so the rest of the pipeline (features -> train -> MLflow)
is runnable immediately.

Usage:
    python src/fetch_data.py --dataset robikscube/hourly-energy-consumption
    python src/fetch_data.py --synthetic   # force synthetic data
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_FILE = RAW_DIR / "energy_weather_raw.csv"


def has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def fetch_from_kaggle(dataset: str) -> pd.DataFrame:
    """Download and load a Kaggle dataset. Requires `pip install kaggle`
    and credentials in ~/.kaggle/kaggle.json (see Kaggle account settings
    -> Create New API Token)."""
    try:
        import kaggle  # noqa: F401 (import triggers auth check)
    except ImportError as e:
        raise RuntimeError(
            "kaggle package not installed. Run: pip install kaggle"
        ) from e

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{dataset}' from Kaggle...")
    api.dataset_download_files(dataset, path=str(RAW_DIR), unzip=True)

    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise RuntimeError(f"No CSV files found after downloading {dataset}")

    print(f"Loading {csvs[0].name}")
    return pd.read_csv(csvs[0])


def generate_synthetic(start="2018-01-01", periods_days=365 * 2, seed=42) -> pd.DataFrame:
    """Synthetic hourly weather + energy demand series.

    Schema mirrors what you'd get merging PJM-style hourly energy load
    with an hourly weather dataset:
        datetime, temp_c, humidity, wind_speed, energy_mw

    Demand is modeled with:
      - a yearly seasonal component (higher in summer/winter, heating/cooling)
      - a daily seasonal component (peak evening, trough overnight)
      - a weekday/weekend effect
      - a temperature-driven component (U-shaped: AC + heating load)
      - autocorrelated noise
    """
    rng = np.random.default_rng(seed)
    n_hours = periods_days * 24
    idx = pd.date_range(start=start, periods=n_hours, freq="h")

    hour = idx.hour.values
    doy = idx.dayofyear.values
    dow = idx.dayofweek.values

    # --- weather ---
    annual_temp_cycle = 15 + 12 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    daily_temp_cycle = 4 * np.sin(2 * np.pi * (hour - 9) / 24)
    temp_noise = rng.normal(0, 2.5, n_hours)
    temp_c = annual_temp_cycle + daily_temp_cycle + temp_noise

    humidity = np.clip(
        60 - 0.8 * (temp_c - 15) + rng.normal(0, 8, n_hours), 15, 100
    )
    wind_speed = np.clip(rng.gamma(shape=2.0, scale=3.0, size=n_hours), 0, None)

    # --- energy demand (MW) ---
    base_load = 2200
    yearly = 300 * np.cos(2 * np.pi * (doy - 15) / 365.25)  # winter peak
    daily = 500 * np.sin(2 * np.pi * (hour - 7) / 24) ** 2 * (
        1 + 0.4 * np.sin(2 * np.pi * (hour - 17) / 24)
    )
    weekday_effect = np.where(dow < 5, 150, -100)
    # U-shaped temperature response: cooling above ~22C, heating below ~10C
    temp_response = 8 * np.clip(temp_c - 22, 0, None) ** 1.3 + 10 * np.clip(
        10 - temp_c, 0, None
    )

    # AR(1) noise for realistic autocorrelation
    ar_noise = np.zeros(n_hours)
    eps = rng.normal(0, 40, n_hours)
    for t in range(1, n_hours):
        ar_noise[t] = 0.85 * ar_noise[t - 1] + eps[t]

    energy_mw = (
        base_load + yearly + daily + weekday_effect + temp_response + ar_noise
    )
    energy_mw = np.clip(energy_mw, 500, None)

    df = pd.DataFrame(
        {
            "datetime": idx,
            "temp_c": temp_c.round(2),
            "humidity": humidity.round(1),
            "wind_speed": wind_speed.round(2),
            "energy_mw": energy_mw.round(1),
        }
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch or synthesize energy+weather data")
    parser.add_argument(
        "--dataset",
        default="robikscube/hourly-energy-consumption",
        help="Kaggle dataset slug (owner/dataset-name)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic data generation, skip Kaggle entirely",
    )
    parser.add_argument("--periods-days", type=int, default=730)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = None
    if not args.synthetic:
        if has_kaggle_credentials():
            try:
                df = fetch_from_kaggle(args.dataset)
            except Exception as e:
                print(f"[warn] Kaggle fetch failed ({e}); falling back to synthetic data.", file=sys.stderr)
        else:
            print(
                "[info] No Kaggle credentials found (~/.kaggle/kaggle.json or "
                "KAGGLE_USERNAME/KAGGLE_KEY env vars). Generating synthetic data instead.\n"
                "        To use real data: pip install kaggle, then place your API "
                "token at ~/.kaggle/kaggle.json and re-run without --synthetic.",
            )

    if df is None:
        df = generate_synthetic(periods_days=args.periods_days)
        print(f"Generated synthetic dataset: {len(df):,} rows")

    df.to_csv(OUT_FILE, index=False)
    print(f"Saved raw data -> {OUT_FILE}  ({len(df):,} rows, {df.shape[1]} cols)")
    print(df.head())


if __name__ == "__main__":
    main()
