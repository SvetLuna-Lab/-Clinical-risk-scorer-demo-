# Clinical Risk Scorer Demo

## Overview (English)

This repository contains a small **end-to-end ML demo** for clinical risk scoring
on tabular patient data.

Core idea:

- we have a tabular dataset of synthetic patients (age, vitals, basic lab values,
  simple flags like `smoker`, `diabetes`, etc.);
- we define a binary target such as **"high complication risk" / "needs hospitalisation"**;
- we build a reproducible ML pipeline in Python:
  - data generation / loading,
  - feature preprocessing (scikit-learn pipeline),
  - model training (Logistic Regression / Random Forest),
  - evaluation (ROC AUC, PR AUC, confusion matrix),
  - simple FastAPI endpoint `/predict` that returns a `risk_score` for one patient.

All configuration (features, model type, paths) is stored in `configs/default.yaml`.
Models and metrics are saved to `models/` and `metrics/` for later inspection.

The dataset is **fully synthetic** and does not contain any real medical data.
The goal of this project is to demonstrate an **engineering-grade ML pipeline**
rather than to build a clinically validated tool.

---

## Обзор (по-русски)

Этот репозиторий — учебный **ML-проект по оценке клинического риска**
на табличных данных.

Идея:

- создаём синтетический датасет пациентов:
  - возраст, базовые показатели, несколько бинарных флагов (`smoker`, `diabetes`, …);
- задаём бинарную цель:
  - например, «высокий риск осложнений / требуется госпитализация»;
- строим сквозной пайплайн:
  - генерация и загрузка данных (`data/`),
  - препроцессинг признаков (скейлинг, One-Hot) через пайплайн scikit-learn,
  - обучение моделей (логистическая регрессия, случайный лес),
  - сохранение артефактов (`models/`) и метрик (`metrics/`),
  - минимальный API на FastAPI (`/predict`), который по JSON-запросу
    возвращает `risk_score` и класс риска.

Проект задуман как демонстрация **инженерного подхода к ML**:
конфигурации в YAML, разделение кода по слоям (`data/`, `features/`, `models/`),
простые автотесты и воспроизводимые эксперименты.

Данные полностью синтетические и не предназначены для реального медицинского применения.
