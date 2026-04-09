# Heart Disease Prediction AI Module

## Overview
This module trains and serves a heart disease risk classifier with calibrated probabilities and a tunable decision threshold. It supports training, inference, and explainability generation for downstream integration.

## Project Structure
- `artifacts/` — trained model, metrics, schema, metadata, threshold, comparisons, and explainability outputs
- `data/` — dataset inputs
- `docs/` — API contract and documentation
- `evaluation/` — explainability generation
- `inference/` — prediction runtime
- `models/` — model utilities
- `training/` — training pipeline

## Final Model
Logistic Regression with calibration and threshold tuning.

## Features Implemented
- Data validation and preprocessing
- Model training and evaluation
- Probability calibration
- Threshold tuning for decisioning
- Model comparison reporting
- Explainability output generation

## Artifacts Generated
- `model.joblib`
- `metrics.json`
- `feature_schema.json`
- `metadata.json`
- `threshold.json`
- `model_comparison.json`
- Explainability outputs

## How to Train
```bash
python -m training.train
```

## How to Run Inference
```bash
python -m inference.predict
```

## How to Generate Explainability Outputs
```bash
python -m evaluation.explain
```

## API Contract Summary
- Endpoint: `POST /predict`
- Request: JSON with required fields `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
- Response: `prediction`, `probability`, `threshold_used`, `success` (or `error` on failure)
- Decision rule: `probability >= threshold` where threshold is loaded from `artifacts/threshold.json`

## Current Status
Training, inference, and explainability pipelines are implemented; artifacts are generated in `artifacts/` after running the training pipeline.
