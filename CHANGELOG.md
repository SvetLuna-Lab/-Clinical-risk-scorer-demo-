# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres (informally) to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2025-11-18

### Added

- Initial end-to-end ML pipeline for clinical risk scoring on synthetic tabular data:
  - `src/data/generate_synthetic.py` — synthetic patient dataset generator
  - `src/data/dataset.py` — train/test loading based on YAML config
  - `src/features/preprocess.py` — preprocessing pipeline (ColumnSelector + StandardScaler)
  - `src/models/train.py` — model training (Logistic Regression) and metrics computation
  - `src/models/evaluate.py` — standalone evaluation script, writing `metrics/metrics.json`

- Centralised path and config handling:
  - `src/paths.py` — project, data, models, metrics directories
  - `configs/default.yaml` — data paths, feature list, target column and model params

- Command-line interface:
  - `src/cli.py` with commands:
    - `generate-data` — generate synthetic train/test datasets
    - `train` — train model and save artifacts
    - `evaluate` — evaluate saved model on the test set

- API layer:
  - `api/app.py` — FastAPI application with:
    - `GET /health` — health check
    - `POST /predict` — risk prediction endpoint returning `risk_score` and `risk_label`

- Testing and notebooks:
  - `tests/test_pipeline_smoke.py` — end-to-end smoke test from data generation to model + metrics artifacts
  - `notebooks/01_exploration.ipynb` — basic EDA of the synthetic dataset

- Documentation and project meta:
  - `README.md` — English and Russian overview of the demo
  - `docs/Overview_RU.md` — detailed Russian technical description of the pipeline
  - `.gitignore`, `pytest.ini`, `requirements.txt`, `requirements-dev.txt` — environment and tooling setup
  - Example `metrics/metrics.json` to illustrate expected metrics output format
