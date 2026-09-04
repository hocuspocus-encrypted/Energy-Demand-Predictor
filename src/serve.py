"""
Day 5: Serve the champion model over a FastAPI /predict endpoint.

Loads from the standalone model/ bundle (see scripts/export_champion.py)
rather than the live MLflow tracking store -- no mlflow.db, mlruns/, or
mlflow package needed at serve time, which keeps the deployed service
small and avoids mlflow's registry-alias resolution bug on Windows.

Run:
    uvicorn src.serve:app --reload
    # docs at http://127.0.0.1:8000/docs
"""
import json
import os
import random
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"


class PredictionRequest(BaseModel):
    temp_c: float
    humidity: float
    wind_speed: float
    hour: int
    dayofweek: int
    month: int
    is_weekend: int
    hour_sin: float
    hour_cos: float
    doy_sin: float
    doy_cos: float
    energy_lag_1h: float
    energy_lag_2h: float
    energy_lag_3h: float
    energy_lag_24h: float
    energy_lag_168h: float
    energy_roll_mean_24h: float
    energy_roll_std_24h: float
    temp_roll_mean_24h: float
    energy_roll_mean_168h: float
    energy_roll_std_168h: float
    temp_roll_mean_168h: float


class PredictionResponse(BaseModel):
    predicted_energy_mw: float
    model_version: str
    run_id: str


class SampleResponse(BaseModel):
    datetime: str
    actual_energy_mw: float
    features: PredictionRequest


def load_bundle():
    pipeline = joblib.load(MODEL_DIR / "pipeline.joblib")
    feature_columns = json.loads(
        (MODEL_DIR / "feature_columns.json").read_text()
    )["feature_columns"]
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    samples = json.loads((MODEL_DIR / "samples.json").read_text())

    expected = set(PredictionRequest.model_fields)
    if set(feature_columns) != expected:
        raise RuntimeError(
            "PredictionRequest fields don't match model/feature_columns.json.\n"
            f"Model expects: {sorted(feature_columns)}\n"
            f"Request has:   {sorted(expected)}\n"
            "Re-run scripts/export_champion.py if the champion changed."
        )

    return pipeline, feature_columns, metadata, samples


app = FastAPI(title="WattFlow")

allowed_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline, _feature_columns, _metadata, _samples = load_bundle()


@app.get("/health")
def health():
    return {"status": "ok", "model_version": _metadata["version"]}


@app.get("/sample", response_model=SampleResponse)
def sample():
    """A real historical row (features + the true energy_mw that occurred),
    for the UI to prefill a prediction form against instead of asking users
    to hand-type 22 engineered feature values."""
    row = random.choice(_samples)
    return SampleResponse(
        datetime=row["datetime"],
        actual_energy_mw=row["actual_energy_mw"],
        features=PredictionRequest(**{c: row[c] for c in _feature_columns}),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    row = pd.DataFrame([request.model_dump()])[_feature_columns]
    try:
        prediction = float(_pipeline.predict(row)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    return PredictionResponse(
        predicted_energy_mw=prediction,
        model_version=str(_metadata["version"]),
        run_id=_metadata["run_id"],
    )
