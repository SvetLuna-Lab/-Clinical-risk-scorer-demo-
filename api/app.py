# api/app.py
"""
FastAPI application for serving the clinical risk scoring model.

Endpoints:
- GET  /health    — simple health check
- POST /predict   — returns risk_score and risk_label for a single patient

Run locally with:
    uvicorn api.app:app --reload
"""

from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.paths import MODELS_DIR


MODEL_PATH = MODELS_DIR / "model_logistic_regression.joblib"


class PatientFeatures(BaseModel):
    """Input schema for one patient."""

    age: int = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Biological sex, encoded as 0/1")
    bmi: float = Field(..., ge=10, le=60, description="Body Mass Index")
    smoker: int = Field(..., ge=0, le=1, description="Smoking status: 0 = no, 1 = yes")
    diabetes: int = Field(..., ge=0, le=1, description="Diabetes: 0 = no, 1 = yes")
    systolic_bp: float = Field(..., ge=60, le=260, description="Systolic blood pressure")
    heart_rate: float = Field(..., ge=30, le=220, description="Heart rate (bpm)")
    cholesterol: float = Field(..., ge=1.0, le=15.0, description="Total cholesterol (synthetic units)")


class PredictionResponse(BaseModel):
    """Output schema for risk prediction."""

    risk_score: float
    risk_label: Literal["low", "high"]
    threshold: float = 0.5
    model_name: str = "model_logistic_regression.joblib"


def load_model():
    """Load the trained sklearn Pipeline from disk."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            f"Train the model first (python -m src.models.train)."
        )
    return joblib.load(MODEL_PATH)


app = FastAPI(title="Clinical Risk Scorer Demo", version="0.1.0")

# Load model at startup
model = load_model()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    """
    Predict clinical risk for a single patient.

    Input: patient features as JSON
    Output: risk_score in [0,1] and discrete risk_label ("low"/"high")
    """
    # Convert Pydantic model to DataFrame with a single row
    features_df = pd.DataFrame([patient.dict()])

    # Predict probability of high risk
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(features_df)[:, 1][0])
    else:
        if hasattr(model, "decision_function"):
            score = float(model.decision_function(features_df)[0])
            proba = float(1 / (1 + np.exp(-score)))
        else:
            # Fallback: use hard prediction only
            label = int(model.predict(features_df)[0])
            proba = float(label)

    threshold = 0.5
    label = "high" if proba >= threshold else "low"

    return PredictionResponse(
        risk_score=proba,
        risk_label=label,
        threshold=threshold,
        model_name=MODEL_PATH.name,
    )
