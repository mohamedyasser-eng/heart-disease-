import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

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

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("=== Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC:      {auc:.4f}")

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


if __name__ == "__main__":
    main()
