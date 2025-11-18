# src/data/generate_synthetic.py
"""
Synthetic patient dataset generation for the demo project.

Outputs:
- data/processed/train.csv
- data/processed/test.csv

Each row represents a single patient with simple tabular features
and a binary target 'high_risk'.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from src.paths import PROCESSED_DATA_DIR, PROJECT_ROOT


def generate_synthetic_patients(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic tabular dataset of patients."""
    rng = np.random.default_rng(seed)

    # Basic features
    patient_id = np.arange(1, n_samples + 1)

    age = rng.integers(18, 90, size=n_samples)
    sex = rng.integers(0, 2, size=n_samples)  # 0/1
    bmi = rng.normal(loc=27.0, scale=4.0, size=n_samples).clip(16, 45)

    smoker = rng.integers(0, 2, size=n_samples)
    diabetes = rng.integers(0, 2, size=n_samples)

    systolic_bp = rng.normal(loc=130, scale=15, size=n_samples).clip(90, 220)
    heart_rate = rng.normal(loc=75, scale=10, size=n_samples).clip(40, 150)
    cholesterol = (
        rng.normal(loc=5.5, scale=1.0, size=n_samples).clip(3.0, 10.0)
    )  # mmol/L, synthetic

    # Simple "hidden" risk formula (for synthetic data only).
    # MUST NOT be used as any kind of medical recommendation.
    z = (
        0.03 * (age - 50)
        + 0.04 * (bmi - 27)
        + 0.5 * smoker
        + 0.7 * diabetes
        + 0.02 * (systolic_bp - 130)
        + 0.01 * (cholesterol - 5.5)
    )

    # Convert to probability via sigmoid
    prob = 1 / (1 + np.exp(-z))

    # Binary target: high_risk
    high_risk = rng.binomial(1, prob)

    df = pd.DataFrame(
        {
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "smoker": smoker,
            "diabetes": diabetes,
            "systolic_bp": systolic_bp,
            "heart_rate": heart_rate,
            "cholesterol": cholesterol,
            "high_risk": high_risk,
        }
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic clinical risk dataset."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Number of synthetic patients to generate (default: 2000).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()

    df = generate_synthetic_patients(n_samples=args.n_samples, seed=args.seed)

    # Train/test split
    n_train = int(len(df) * args.train_ratio)
    df_shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    train_df = df_shuffled.iloc[:n_train].copy()
    test_df = df_shuffled.iloc[n_train:].copy()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[OK] Generated train dataset: {train_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] Generated test dataset:  {test_path.relative_to(PROJECT_ROOT)}")
    print(f"Train size: {len(train_df)}, test size: {len(test_df)}")


if __name__ == "__main__":
    main()
