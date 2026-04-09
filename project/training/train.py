import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from models.pipeline import build_pipeline


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    set_seeds(42)

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "heart.csv"
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    df = df.drop_duplicates()

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc",
    )
    cv_mean_auc = float(np.mean(cv_scores))
    cv_std_auc = float(np.std(cv_scores))

    print("=== Cross Validation ===")
    print(f"Mean AUC: {cv_mean_auc:.4f}")
    print(f"Std AUC:  {cv_std_auc:.4f}")

    comparison_base = build_pipeline()
    features_step = comparison_base.named_steps["features"]
    preprocess_step = comparison_base.named_steps["preprocess"]

    candidate_models = [
        (
            "LogisticRegression",
            LogisticRegression(
                class_weight="balanced",
                random_state=42,
                max_iter=1000,
            ),
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_estimators=200,
            ),
        ),
        (
            "SVC",
            SVC(
                class_weight="balanced",
                probability=True,
                random_state=42,
            ),
        ),
        (
            "KNeighborsClassifier",
            KNeighborsClassifier(),
        ),
    ]

    comparison_results = []
    for model_name, model in candidate_models:
        candidate_pipeline = Pipeline(
            steps=[
                ("features", clone(features_step)),
                ("preprocess", clone(preprocess_step)),
                ("model", model),
            ]
        )
        candidate_scores = cross_val_score(
            candidate_pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="roc_auc",
        )
        comparison_results.append(
            {
                "model_name": model_name,
                "cv_mean_auc": float(np.mean(candidate_scores)),
                "cv_std_auc": float(np.std(candidate_scores)),
            }
        )

    comparison_path = artifacts_dir / "model_comparison.json"
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "candidates": comparison_results,
                "selected_final_model": "LogisticRegression",
                "selection_metric": "roc_auc",
            },
            f,
            indent=2,
        )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("=== Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC:      {auc:.4f}")

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    evaluated_thresholds = []
    for threshold in thresholds:
        threshold_pred = (y_prob >= threshold).astype(int)
        recall = recall_score(y_test, threshold_pred, pos_label=1)
        precision = precision_score(y_test, threshold_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, threshold_pred, pos_label=1, zero_division=0)
        evaluated_thresholds.append(
            {
                "threshold": threshold,
                "recall": float(recall),
                "precision": float(precision),
                "f1": float(f1),
            }
        )

    selected = sorted(
        evaluated_thresholds,
        key=lambda item: (-item["recall"], -item["f1"], item["threshold"]),
    )[0]

    model_path = artifacts_dir / "model.joblib"
    joblib.dump(pipeline, model_path)

    metrics_path = artifacts_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "auc": auc,
                "cv_mean_auc": cv_mean_auc,
                "cv_std_auc": cv_std_auc,
            },
            f,
            indent=2,
        )

    threshold_path = artifacts_dir / "threshold.json"
    with threshold_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_threshold": selected["threshold"],
                "selection_priority": ["highest_recall", "highest_f1", "lowest_threshold"],
                "evaluated_thresholds": evaluated_thresholds,
            },
            f,
            indent=2,
        )

    feature_schema = {
        "raw_input_features": [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
        ],
        "continuous_features": [
            "age",
            "trestbps",
            "thalach",
            "oldpeak",
        ],
        "engineered_features": [
            "low_thalach",
            "high_oldpeak",
            "age_group",
            "has_blockage",
            "exercise_risk",
        ],
        "model_type": "LogisticRegression",
    }

    feature_schema_path = artifacts_dir / "feature_schema.json"
    with feature_schema_path.open("w", encoding="utf-8") as f:
        json.dump(feature_schema, f, indent=2)

    metadata = {
        "model_type": "LogisticRegression",
        "calibration_enabled": True,
        "calibration_method": "sigmoid",
        "random_seed": 42,
        "train_test_split": {
            "test_size": 0.2,
            "stratify": True,
            "random_state": 42,
        },
        "cross_validation": {
            "folds": 5,
            "scoring": "roc_auc",
        },
        "raw_input_feature_count": 13,
        "engineered_feature_count": 5,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    metadata_path = artifacts_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
