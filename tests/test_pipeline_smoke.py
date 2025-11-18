# tests/test_pipeline_smoke.py
"""
End-to-end smoke test for the clinical risk scoring pipeline.

Steps:
- generate a small synthetic dataset (train/test CSV)
- run training + evaluation using the default config
- check that metrics look sane and artifacts (model + metrics) are saved
"""

from pathlib import Path

import pandas as pd

from src.data.generate_synthetic import generate_synthetic_patients
from src.models.train import train_and_evaluate
from src.paths import PROCESSED_DATA_DIR, MODELS_DIR, METRICS_DIR


def test_pipeline_smoke():
    # 1) Generate a small synthetic dataset
    n_samples = 500
    seed = 123

    df = generate_synthetic_patients(n_samples=n_samples, seed=seed)

    # Simple train/test split (80/20) with shuffling
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train = int(0.8 * n_samples)

    train_df = df_shuffled.iloc[:n_train].copy()
    test_df = df_shuffled.iloc[n_train:].copy()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 2) Run training + evaluation using the default config
    metrics = train_and_evaluate(Path("default.yaml"))

    # 3) Basic sanity checks on metrics
    assert "accuracy" in metrics, "Accuracy metric is missing"
    assert 0.0 <= metrics["accuracy"] <= 1.0, "Accuracy must be in [0, 1]"

    # Optional: check that ROC AUC and AP are in a valid range if not NaN
    if "roc_auc" in metrics and not pd.isna(metrics["roc_auc"]):
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    if "average_precision" in metrics and not pd.isna(metrics["average_precision"]):
        assert 0.0 <= metrics["average_precision"] <= 1.0

    # 4) Check that artifacts were saved
    model_files = list(MODELS_DIR.glob("model_*.joblib"))
    metrics_files = list(METRICS_DIR.glob("metrics_*.json"))

    assert model_files, "No model artifacts were saved in models/ directory"
    assert metrics_files, "No metrics artifacts were saved in metrics/ directory"
