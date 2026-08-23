"""
Day 1-2: Baseline training run with local MLflow experiment tracking.

Trains a scikit-learn model on the engineered features and logs
hyperparameters, metrics, and the fitted pipeline (preprocessing +
model) as an MLflow run, so every run is reproducible from the
tracking store alone.

Run:
    mlflow ui --backend-store-uri ./mlruns   # in one terminal, to view results
    python src/train.py                      # in another, to train + log
"""
import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.pipeline import Pipeline

from features import PROCESSED_FILE, TARGET, build_feature_frame, build_preprocessing_pipeline, get_feature_columns

ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = ROOT / "mlruns"


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split -- never shuffle time-series data."""
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--experiment-name", default="energy-demand-forecasting")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(args.experiment_name)

    # Rebuild features fresh each run (cheap here; keeps the run fully
    # reproducible from raw data rather than depending on a stale CSV).
    df = build_feature_frame()
    feat_cols = get_feature_columns(df)
    train_df, test_df = time_split(df, test_frac=args.test_frac)

    X_train, y_train = train_df[feat_cols], train_df[TARGET]
    X_test, y_test = test_df[feat_cols], test_df[TARGET]

    preprocessing = build_preprocessing_pipeline(feat_cols)
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=-1,
        random_state=42,
    )
    pipeline = Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "model_type": "RandomForestRegressor",
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf,
                "n_features": len(feat_cols),
                "n_train_rows": len(train_df),
                "n_test_rows": len(test_df),
                "test_frac": args.test_frac,
                "random_state": 42,
            }
        )
        mlflow.log_dict({"feature_columns": feat_cols}, "feature_columns.json")

        pipeline.fit(X_train, y_train)

        train_metrics = evaluate(y_train, pipeline.predict(X_train))
        test_metrics = evaluate(y_test, pipeline.predict(X_test))

        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        mlflow.sklearn.log_model(pipeline, name="model")

        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
        print("Train metrics:", {k: round(v, 3) for k, v in train_metrics.items()})
        print("Test metrics: ", {k: round(v, 3) for k, v in test_metrics.items()})


if __name__ == "__main__":
    main()
