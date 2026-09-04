"""
Day 3-4: Champion/challenger retraining.

Trains a new model (via src/train.py's run_training()), registers it as a
new version of the "wattflow" MLflow registered model, and
compares its test MAE against the current @champion version. The new
version is only promoted to @champion if it has a lower test MAE than the
existing champion (or if there is no champion yet).

Writes data/retrain_status.json with the outcome so it can be picked up by
the Airflow DAG's report_outcome task (or inspected manually).

Usage:
    python scripts/retrain_pipeline.py [--n-estimators 300] [--max-depth 12] ...
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train import run_training  # noqa: E402

REGISTERED_MODEL_NAME = "wattflow"
CHAMPION_ALIAS = "champion"
STATUS_FILE = ROOT / "data" / "retrain_status.json"


def get_champion_test_mae(client: MlflowClient) -> tuple[str | None, float | None]:
    """Returns (run_id, test_mae) of the current champion, or (None, None)
    if no champion alias has been set yet."""
    try:
        champion_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return None, None

    run = client.get_run(champion_version.run_id)
    return champion_version.run_id, run.data.metrics.get("test_mae")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--experiment-name", default="energy-demand-forecasting")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    run_name = args.run_name or f"retrain_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"

    result = run_training(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        test_frac=args.test_frac,
        experiment_name=args.experiment_name,
        run_name=run_name,
    )
    new_run_id = result["run_id"]
    new_test_mae = result["test_metrics"]["mae"]

    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

    try:
        client.get_registered_model(REGISTERED_MODEL_NAME)
    except MlflowException:
        client.create_registered_model(REGISTERED_MODEL_NAME)

    new_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=result["model_uri"],
        run_id=new_run_id,
    )

    champion_run_id, champion_test_mae = get_champion_test_mae(client)

    promoted = champion_test_mae is None or new_test_mae < champion_test_mae
    if promoted:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, new_version.version)

    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": new_run_id,
        "model_version": new_version.version,
        "new_test_mae": new_test_mae,
        "champion_test_mae": champion_test_mae,
        "champion_run_id": champion_run_id if not promoted else new_run_id,
        "promoted": promoted,
    }

    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2))

    if promoted:
        prior = "no prior champion" if champion_test_mae is None else f"prior champion test_mae={champion_test_mae:.3f}"
        print(f"[PROMOTED] version {new_version.version} (test_mae={new_test_mae:.3f}) is the new champion ({prior}).")
    else:
        print(
            f"[NOT PROMOTED] version {new_version.version} (test_mae={new_test_mae:.3f}) "
            f"did not beat champion (test_mae={champion_test_mae:.3f})."
        )


if __name__ == "__main__":
    main()
