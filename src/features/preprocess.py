# src/features/preprocess.py
"""
Feature preprocessing utilities.

For this demo project all features are numeric, so we apply a simple
StandardScaler over the selected feature columns.

If you later add categorical features, this module is a good place
to introduce a ColumnTransformer with OneHotEncoder etc.
"""

from typing import List

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_preprocessing_pipeline(feature_columns: List[str]) -> Pipeline:
    """
    Build a preprocessing pipeline for the given feature columns.

    Currently:
    - selects the feature columns from the input DataFrame
    - applies StandardScaler to all of them

    Returns:
        sklearn Pipeline that can be used as the first step in a full
        model pipeline: (preprocess -> model).
    """

    class ColumnSelector(TransformerMixin):
        """Simple transformer to select a subset of columns by name."""

        def __init__(self, columns: List[str]):
            self.columns = list(columns)

        def fit(self, X: pd.DataFrame, y=None):
            return self

        def transform(self, X: pd.DataFrame):
            return X[self.columns]

    preprocessing = Pipeline(
        steps=[
            ("select_features", ColumnSelector(feature_columns)),
            ("scale", StandardScaler()),
        ]
    )

    return preprocessing
