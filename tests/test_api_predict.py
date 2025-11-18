# tests/test_api_predict.py
"""
Smoke test for the FastAPI /predict endpoint.

Workflow:
- generate a small synthetic dataset
- train a model using the default config
- import FastAPI app and call /health and /predict
"""

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.data.generate_synthetic import generate_synthetic_patients
from src.models.train import train_and_evaluate
from src.paths import PROCESSED_DATA_DIR, PROJECT_ROOT

# 1) Prepare data and train model BEFORE importing the FastAPI app


def _prepare_data_and_model():
    n_samples = 400
    seed = 999

    df = generate_synthetic_patients(n_samples=n_samples, seed=seed)

    # Simple shuffled train/test split
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train = int(0.8 * n_samples)

    train_df = df_shuffled.iloc[:n_train].copy()
    test_df = df_shuffled.iloc[n_train:].copy()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Train model using default config
    metrics = train_and_evaluate(Path("default.yaml"))
    return train_df, metrics


# Generate data and train model once at import time for this test module
_TRAIN_DF, _METRICS = _prepare_data_and_model()

# 2) Only now import the FastAPI app (model file already exists)
from api.app import app  # noqa: E402


client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"


def test_predict_endpoint():
    # Use the first row from training data as a template for the request
    row = _TRAIN_DF.iloc[0]

    # Features expected by PatientFeatures in api/app.py
    payload = {
        "age": int(row["age"]),
        "sex": int(row["sex"]),
        "bmi": float(row["bmi"]),
        "smoker": int(row["smoker"]),
        "diabetes": int(row["diabetes"]),
        "systolic_bp": float(row["systolic_bp"]),
        "heart_rate": float(row["heart_rate"]),
        "cholesterol": float(row["cholesterol"]),
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert "risk_score" in data
    assert "risk_label" in data
    assert "threshold" in data

    risk_score = data["risk_score"]
    assert 0.0 <= risk_score <= 1.0, "risk_score must be in [0, 1]"
    assert data["risk_label"] in {"low", "high"}
