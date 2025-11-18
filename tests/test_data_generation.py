# tests/test_data_generation.py
"""
Unit test for the synthetic data generator.

Checks:
- correct number of rows
- required columns are present
- high_risk is binary and has no missing values
"""

import numpy as np

from src.data.generate_synthetic import generate_synthetic_patients


def test_generate_synthetic_patients_basic():
    n_samples = 200
    seed = 123

    df = generate_synthetic_patients(n_samples=n_samples, seed=seed)

    # Shape
    assert len(df) == n_samples, "Unexpected number of generated rows"

    # Required columns
    expected_columns = {
        "patient_id",
        "age",
        "sex",
        "bmi",
        "smoker",
        "diabetes",
        "systolic_bp",
        "heart_rate",
        "cholesterol",
        "high_risk",
    }
    assert expected_columns.issubset(
        set(df.columns)
    ), "Generated dataframe is missing some expected columns"

    # high_risk must be binary and non-null
    assert df["high_risk"].isna().sum() == 0, "high_risk contains NaN values"
    unique_values = set(df["high_risk"].unique())
    assert unique_values.issubset({0, 1}), f"high_risk has non-binary values: {unique_values}"

    # Basic sanity: there should be at least some positives and some negatives
    positives = (df["high_risk"] == 1).sum()
    negatives = (df["high_risk"] == 0).sum()
    assert positives > 0, "No positive high_risk examples generated"
    assert negatives > 0, "No negative high_risk examples generated"
