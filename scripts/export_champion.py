"""
Day 5: Export the current @champion model to a standalone, mlflow-free
bundle for deployment.

The FastAPI service (src/serve.py) loads from this bundle rather than the
live MLflow tracking store: it's much smaller than shipping the full
mlruns/ history (hundreds of MB across every run), doesn't need mlflow.db
or a tracking URI in production, and sidesteps mlflow's registry-alias
resolution bug on Windows (see src/serve.py's comment on runs:/ vs
models:/@alias).

Re-run this any time scripts/retrain_pipeline.py promotes a new champion,
then commit the updated model/ directory.

Usage:
    python scripts/export_champion.py
"""
import json
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
REGISTERED_MODEL_NAME = "wattflow"
CHAMPION_ALIAS = "champion"
MODEL_DIR = ROOT / "model"


def main():
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    client = MlflowClient()
    mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)

    # runs:/ rather than models:/@alias -- see src/serve.py's load_champion
    # docstring for why the alias-URI form breaks on Windows.
    pipeline = mlflow.sklearn.load_model(f"runs:/{mv.run_id}/model")
    feature_columns = mlflow.artifacts.load_dict(
        f"runs:/{mv.run_id}/feature_columns.json"
    )["feature_columns"]
    run = client.get_run(mv.run_id)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "pipeline.joblib", compress=3)
    (MODEL_DIR / "feature_columns.json").write_text(
        json.dumps({"feature_columns": feature_columns}, indent=2)
    )
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": mv.run_id,
                "version": mv.version,
                "test_metrics": {
                    k.removeprefix("test_"): v
                    for k, v in run.data.metrics.items()
                    if k.startswith("test_")
                },
            },
            indent=2,
        )
    )
    print(f"Exported champion v{mv.version} (run {mv.run_id}) -> {MODEL_DIR}/")


if __name__ == "__main__":
    main()
