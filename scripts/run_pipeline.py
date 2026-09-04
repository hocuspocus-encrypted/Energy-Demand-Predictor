"""
Day 6-7: Run the full pipeline end to end in one process.

fetch_data -> validate_data -> build_features -> retrain (champion/challenger)

The same steps the Airflow DAG (dags/energy_demand_pipeline.py) runs on a
schedule, callable directly for a one-off containerized or local run. Each
step shells out to the existing standalone script, so this stays a thin
wrapper over code that's also runnable by hand -- same design as the DAG.

Usage:
    python scripts/run_pipeline.py                          # synthetic data
    python scripts/run_pipeline.py --dataset robikscube/hourly-energy-consumption --kaggle-file PJME_hourly.csv
    python scripts/run_pipeline.py --n-estimators 500 --max-depth 20
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(*args: str) -> None:
    print(f"\n$ python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run the full fetch->validate->features->retrain pipeline")
    parser.add_argument("--dataset", default="robikscube/hourly-energy-consumption")
    parser.add_argument("--kaggle-file", default="PJME_hourly.csv")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data, skip Kaggle")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    fetch_args = ["src/fetch_data.py"]
    if args.synthetic:
        fetch_args.append("--synthetic")
    else:
        fetch_args += ["--dataset", args.dataset, "--kaggle-file", args.kaggle_file]
    run_script(*fetch_args)

    run_script("scripts/validate_data.py")
    run_script("src/features.py")

    retrain_args = [
        "scripts/retrain_pipeline.py",
        "--n-estimators", str(args.n_estimators),
        "--max-depth", str(args.max_depth),
        "--min-samples-leaf", str(args.min_samples_leaf),
    ]
    if args.run_name:
        retrain_args += ["--run-name", args.run_name]
    run_script(*retrain_args)

    print("\nPipeline complete. Run scripts/export_champion.py to update the serving bundle.")


if __name__ == "__main__":
    main()
