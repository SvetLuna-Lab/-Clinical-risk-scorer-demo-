# src/models/evaluate.py
"""
Standalone evaluation script for the Clinical Risk Scorer demo.

Use case:
- you already have a trained model saved in models/
- you want to re-evaluate it on the current test set
  defined in the YAML config, and write metrics to metrics/metrics.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
)
import joblib

from src.data.dataset import load_train_test_from_config, load_config
from src.paths import CONFIGS_DIR, MODELS_DIR, METRICS_DIR, PROJECT_ROOT


def evaluate_model(
    config_path: Path,
    model_path: Path,
    output_path: Path | None = None,
) -> Dict[str, float]:
    """
    Load a trained model and evaluate it on the test set defined in the config.

    Args:
        config_path: path to YAML config
        model_path: path to the trained sklearn Pipeline (.joblib)
        output_path: where to save JSON metrics (defaults to metrics/metrics.json)

    Returns:
        dict with metrics: roc_auc, average_precision, accuracy, f1
    """
    # Resolve config path
    if not config_path.is_absolute():
        config_path = (CONFIGS_DIR / config_path).resolve() if not config_path.exists() else config_path

    cfg = load_config(config_path)

    # Load data according to config
    df_train, df_test, feature_columns, target_column = load_train_test_from_config(
        config_path
    )

    X_test = df_test[feature_columns]
    y_test = df_test[target_column].astype(int)

    # Resolve model path
    if not model_path.is_absolute():
        model_path = (MODELS_DIR / model_path).resolve() if not model_path.exists() else model_path

    model = joblib.load(model_path)

    # Predict probabilities and labels
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-scores))
    else:
        # As a last resort, use predicted labels only
        y_proba = model.predict(X_test)

    y_pred = model.predict(X_test)

    # Compute metrics
    metrics: Dict[str, float] = {}

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["average_precision"] = float(average_precision_score(y_test, y_proba))
    except Exception:
        metrics["average_precision"] = float("nan")

    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
    metrics["f1"] = float(f1_score(y_test, y_pred))

    # Save metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = METRICS_DIR / "metrics.json"
    elif not output_path.is_absolute():
        output_path = (METRICS_DIR / output_path).resolve()

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Saved evaluation metrics to: {output_path.relative_to(PROJECT_ROOT)}")
    print("Metrics:", metrics)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained clinical risk scoring model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default.yaml",
        help="Config file name or path (default: default.yaml in configs/).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="model_logistic_regression.joblib",
        help="Model file name or path (default: model_logistic_regression.joblib in models/).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics.json",
        help="Output JSON file name for metrics (default: metrics.json in metrics/).",
    )

    args = parser.parse_args()

    evaluate_model(
        config_path=Path(args.config),
        model_path=Path(args.model_path),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
