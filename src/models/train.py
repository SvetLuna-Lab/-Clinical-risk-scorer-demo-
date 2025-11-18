# src/models/train.py
"""
Model training script for the Clinical Risk Scorer demo.

Steps:
- load config (YAML)
- load train/test datasets
- build preprocessing + model pipeline
- fit on train
- evaluate on test
- save the trained pipeline and metrics to disk
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
import joblib

from src.data.dataset import load_train_test_from_config, load_config
from src.features.preprocess import build_preprocessing_pipeline
from src.paths import CONFIGS_DIR, MODELS_DIR, METRICS_DIR, PROJECT_ROOT


def build_model(model_cfg: Dict[str, Any]) -> Pipeline:
    """
    Build a full sklearn Pipeline: preprocessing + model.

    Currently supports:
    - type: "logistic_regression"
    """
    model_type = model_cfg.get("type", "logistic_regression")
    params = model_cfg.get("params", {})

    if model_type == "logistic_regression":
        estimator = LogisticRegression(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Preprocessing will be plugged in later, once we know feature columns
    # (we build it in train_and_evaluate).
    # Here we only build the "model" part.
    return estimator  # returned estimator, not full pipeline


def train_and_evaluate(config_path: Path) -> Dict[str, float]:
    """
    Train the model defined in the config and evaluate it on the test set.

    Returns:
        metrics_dict with keys: roc_auc, average_precision, accuracy, f1
    """
    # Load config
    if not config_path.is_absolute():
        config_path = (CONFIGS_DIR / config_path).resolve() if not config_path.exists() else config_path

    cfg = load_config(config_path)

    # Load data according to config
    df_train, df_test, feature_columns, target_column = load_train_test_from_config(
        config_path
    )

    X_train = df_train[feature_columns]
    y_train = df_train[target_column].astype(int)

    X_test = df_test[feature_columns]
    y_test = df_test[target_column].astype(int)

    # Build preprocessing + model pipeline
    preprocessing = build_preprocessing_pipeline(feature_columns)
    base_model = build_model(cfg["model"])

    model_type = cfg["model"].get("type", "logistic_regression")
    full_pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("model", base_model),
        ]
    )

    # Fit
    full_pipeline.fit(X_train, y_train)

    # Predict probabilities and labels on test
    if hasattr(full_pipeline, "predict_proba"):
        y_proba = full_pipeline.predict_proba(X_test)[:, 1]
    else:
        # Fallback: use decision_function and convert to [0,1] via sigmoid
        if hasattr(full_pipeline, "decision_function"):
            scores = full_pipeline.decision_function(X_test)
            y_proba = 1 / (1 + np.exp(-scores))
        else:
            # As a last resort, use predicted labels only
            y_proba = full_pipeline.predict(X_test)

    y_pred = full_pipeline.predict(X_test)

    # Compute basic metrics
    metrics_dict: Dict[str, float] = {}

    try:
        metrics_dict["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        metrics_dict["roc_auc"] = float("nan")

    try:
        metrics_dict["average_precision"] = float(average_precision_score(y_test, y_proba))
    except Exception:
        metrics_dict["average_precision"] = float("nan")

    metrics_dict["accuracy"] = float(accuracy_score(y_test, y_pred))
    metrics_dict["f1"] = float(f1_score(y_test, y_pred))

    # Save model pipeline
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"model_{model_type}.joblib"
    joblib.dump(full_pipeline, model_path)

    # Save metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / f"metrics_{model_type}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"[OK] Saved model:   {model_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] Saved metrics: {metrics_path.relative_to(PROJECT_ROOT)}")
    print("Metrics:", metrics_dict)

    return metrics_dict


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the clinical risk scoring model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default.yaml",
        help="Config file name or path (default: default.yaml in configs/).",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    train_and_evaluate(config_path)


if __name__ == "__main__":
    main()
