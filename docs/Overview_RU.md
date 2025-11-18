# Clinical Risk Scorer Demo — обзор проекта (RU)

## 1. Идея проекта

Этот репозиторий — учебный пример **ML-пайплайна для оценки клинического риска**
на табличных данных.

Особенности:

- данные **полностью синтетические** (случайно сгенерированные пациенты);
- есть понятная бинарная цель `high_risk` (0/1);
- построен воспроизводимый пайплайн:
  - генерация и загрузка данных;
  - препроцессинг признаков;
  - обучение модели;
  - вычисление метрик;
  - простой API на FastAPI (`/predict`).

Цель проекта — показать тебя как инженера, который умеет:

- структурировать код (`src/data`, `src/features`, `src/models`, `api`);
- работать с конфигами (YAML);
- собирать end-to-end ML-систему «данные → модель → метрики → сервис».

Проект **не предназначен** для реального медицинского применения и не может
использоваться как клинический инструмент.

---

## 2. Структура проекта

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

