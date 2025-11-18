# src/data/dataset.py
"""
Utilities for loading train/test datasets based on a YAML config.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any

import pandas as pd
import yaml

from src.paths import PROJECT_ROOT, CONFIGS_DIR


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load a YAML config file and return it as a Python dict.
    """
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_train_test_from_config(
    config_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], str]:
    """
    Load train and test datasets according to paths specified in the config.

    Returns:
        df_train: training DataFrame
        df_test: test DataFrame
        feature_columns: list of feature column names
        target_column: name of the target column
    """
    # Resolve config path relative to project root if needed
    if not config_path.is_absolute():
        config_path = (CONFIGS_DIR / config_path).resolve() if not config_path.exists() else config_path

    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    train_rel = Path(data_cfg["train_path"])
    test_rel = Path(data_cfg["test_path"])

    train_path = PROJECT_ROOT / train_rel
    test_path = PROJECT_ROOT / test_rel

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    feature_columns = list(data_cfg["feature_columns"])
    target_column = data_cfg["target_column"]

    return df_train, df_test, feature_columns, target_column
