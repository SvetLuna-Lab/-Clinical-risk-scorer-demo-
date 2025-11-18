# src/paths.py
from pathlib import Path

"""
Centralised definition of important project paths.
"""

# Project root (the "clinical-risk-scorer-demo" folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Configs
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Models and metrics
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"

# Ensure that all base directories exist
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, METRICS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
