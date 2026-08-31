"""
Day 3-4: Data validation gate.

Runs against the raw fetched data (before feature engineering) and
separates problems into two tiers:

- Blocking checks: bad enough that downstream training would be
  wrong or would crash outright. Any failure here exits non-zero,
  which fails the DAG task and stops the pipeline before it wastes
  a training run on bad data.
- Warning checks: suspicious but not fatal (a few outliers, a small
  gap in the series). Logged so a human can look, but don't block
  the run.

Usage:
    python scripts/validate_data.py [--input path/to/raw.csv]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "energy_weather_raw.csv"

REQUIRED_NUMERIC_COLUMNS = ["temp_c", "humidity", "wind_speed", "energy_mw"]
TARGET = "energy_mw"

TEMP_RANGE = (-50, 55)          # deg C, plausible surface air temp range
HUMIDITY_RANGE = (0, 100)       # percent


def load(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    return df


def check_schema(df: pd.DataFrame) -> list[str]:
    errors = []
    required = {"datetime", *REQUIRED_NUMERIC_COLUMNS}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"missing required columns: {sorted(missing)}")
        return errors  # can't check types on columns that don't exist

    parsed_dt = pd.to_datetime(df["datetime"], errors="coerce")
    if parsed_dt.isna().any():
        errors.append(f"{parsed_dt.isna().sum()} row(s) have an unparseable 'datetime' value")

    for col in REQUIRED_NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"column '{col}' is not numeric (dtype={df[col].dtype})")

    return errors


def check_duplicate_timestamps(df: pd.DataFrame) -> list[str]:
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    dupes = dt.duplicated().sum()
    if dupes:
        return [f"{dupes} duplicate 'datetime' value(s) found"]
    return []


def check_target_nulls(df: pd.DataFrame) -> list[str]:
    nulls = df[TARGET].isna().sum()
    if nulls:
        return [f"{nulls} null value(s) in target column '{TARGET}'"]
    return []


def check_out_of_range(df: pd.DataFrame) -> list[str]:
    warnings = []

    lo, hi = TEMP_RANGE
    bad_temp = ((df["temp_c"] < lo) | (df["temp_c"] > hi)).sum()
    if bad_temp:
        warnings.append(f"{bad_temp} row(s) have temp_c outside plausible range [{lo}, {hi}]")

    lo, hi = HUMIDITY_RANGE
    bad_humidity = ((df["humidity"] < lo) | (df["humidity"] > hi)).sum()
    if bad_humidity:
        warnings.append(f"{bad_humidity} row(s) have humidity outside [{lo}, {hi}]")

    bad_wind = (df["wind_speed"] < 0).sum()
    if bad_wind:
        warnings.append(f"{bad_wind} row(s) have negative wind_speed")

    bad_energy = (df[TARGET] < 0).sum()
    if bad_energy:
        warnings.append(f"{bad_energy} row(s) have negative {TARGET}")

    return warnings


def check_time_gaps(df: pd.DataFrame) -> list[str]:
    dt = pd.to_datetime(df["datetime"], errors="coerce").dropna().sort_values()
    if len(dt) < 2:
        return []
    diffs = dt.diff().dropna()
    gaps = diffs[diffs != pd.Timedelta(hours=1)]
    if len(gaps):
        return [f"{len(gaps)} gap(s) in the hourly time series (expected exactly 1h between rows)"]
    return []


def validate(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    blocking = check_schema(df)
    if blocking:
        # schema is broken enough that the other checks would raise
        return blocking, []

    blocking += check_duplicate_timestamps(df)
    blocking += check_target_nulls(df)

    warnings = check_out_of_range(df)
    warnings += check_time_gaps(df)

    return blocking, warnings


def main():
    parser = argparse.ArgumentParser(description="Validate raw energy/weather data before training")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    print(f"Validating {args.input}")
    df = load(args.input)
    blocking, warnings = validate(df)

    if warnings:
        print(f"\n[WARN] {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")

    if blocking:
        print(f"\n[FAIL] {len(blocking)} blocking error(s):")
        for e in blocking:
            print(f"  - {e}")
        sys.exit(1)

    print(f"\n[OK] {len(df):,} rows passed all blocking checks.")


if __name__ == "__main__":
    main()
