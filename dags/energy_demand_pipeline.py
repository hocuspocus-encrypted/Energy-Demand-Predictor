"""
Day 3-4: Orchestration.

Runs the full pipeline daily: fetch data -> validate -> build features ->
retrain (champion/challenger) -> report outcome. Each task shells out to
the existing standalone scripts under src/ and scripts/, so the DAG stays
a thin scheduling layer over code that's also runnable by hand.

Uses Airflow 3.x's airflow.sdk import surface (airflow.decorators is the
2.x path and is deprecated in 3.x).
"""
import json
import subprocess
from datetime import datetime
from pathlib import Path

from airflow.sdk import dag, task

PROJECT_ROOT = Path("/opt/airflow/project")


def run_script(*args: str) -> None:
    subprocess.run(["python", *args], check=True, cwd=str(PROJECT_ROOT))


@dag(
    dag_id="energy_demand_pipeline",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["energy-demand", "days-3-4"],
)
def energy_demand_pipeline():
    @task
    def fetch_data():
        run_script("src/fetch_data.py", "--synthetic")

    @task
    def validate_data():
        run_script("scripts/validate_data.py")

    @task
    def build_features():
        run_script("src/features.py")

    @task
    def retrain_model():
        run_script("scripts/retrain_pipeline.py")

    @task
    def report_outcome():
        status_path = PROJECT_ROOT / "data" / "retrain_status.json"
        status = json.loads(status_path.read_text())
        if status["promoted"]:
            print(
                f"Promoted model version {status['model_version']} "
                f"(run {status['run_id']}) as new champion: "
                f"test_mae={status['new_test_mae']:.3f}"
            )
        else:
            print(
                f"Challenger (run {status['run_id']}, "
                f"test_mae={status['new_test_mae']:.3f}) did not beat "
                f"champion (test_mae={status['champion_test_mae']:.3f}); "
                "champion unchanged."
            )
        return status

    fetch_data() >> validate_data() >> build_features() >> retrain_model() >> report_outcome()


energy_demand_pipeline()
