# Clinical Risk Scorer Demo — Project Overview (EN)

## 1. Idea

This repository is a small **end-to-end ML demo** for clinical risk scoring
on tabular patient data.

Key goals:

- show a clean, reproducible ML pipeline in Python;
- keep the dataset fully **synthetic** (no real medical data);
- provide a minimal but realistic structure:
  - data generation and loading,
  - feature preprocessing,
  - model training and evaluation,
  - simple FastAPI service for `/predict`.

The project is **not** intended for real clinical use and must not be treated
as a medical device or decision-support tool. It is an engineering demo only.

---

## 2. Repository structure

```text
clinical-risk-scorer-demo/
├─ data/
│  ├─ raw/                      # optional, for raw CSVs or experiments
│  └─ processed/                # train.csv, test.csv
├─ configs/
│  └─ default.yaml              # config: data paths, features, target, model params
├─ src/
│  ├─ __init__.py
│  ├─ paths.py                  # centralised project/data/models/metrics paths
│  ├─ data/
│  │  ├─ generate_synthetic.py  # synthetic patient generation
│  │  └─ dataset.py             # load train/test according to YAML config
│  ├─ features/
│  │  └─ preprocess.py          # sklearn preprocessing pipeline
│  ├─ models/
│  │  ├─ train.py               # train model, compute metrics, save artifacts
│  │  └─ evaluate.py            # standalone evaluation, write metrics.json
│  └─ cli.py                    # CLI with commands: generate-data, train, evaluate
├─ api/
│  └─ app.py                    # FastAPI app exposing /health and /predict
├─ notebooks/
│  └─ 01_exploration.ipynb      # basic EDA of the synthetic dataset
├─ tests/
│  ├─ test_data_generation.py   # unit test for synthetic data generator
│  ├─ test_pipeline_smoke.py    # end-to-end pipeline smoke test
│  └─ test_api_predict.py       # smoke test for /predict endpoint
├─ models/                      # trained sklearn pipelines (.joblib)
├─ metrics/                     # metrics JSON files
├─ docs/
│  ├─ Overview_RU.md            # Russian technical overview
│  └─ Overview_EN.md            # (this file) English technical overview
├─ README.md
├─ CHANGELOG.md
├─ LICENSE
├─ requirements.txt
├─ requirements-dev.txt
├─ .gitignore
└─ pytest.ini
