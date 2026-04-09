import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REQUIRED_FIELDS = {
    "age": (int, float, np.integer, np.floating),
    "sex": (int, np.integer),
    "cp": (int, np.integer),
    "trestbps": (int, float, np.integer, np.floating),
    "chol": (int, float, np.integer, np.floating),
    "fbs": (int, np.integer),
    "restecg": (int, np.integer),
    "thalach": (int, float, np.integer, np.floating),
    "exang": (int, np.integer),
    "oldpeak": (int, float, np.integer, np.floating),
    "slope": (int, np.integer),
    "ca": (int, np.integer),
    "thal": (int, np.integer),
}


def load_model():
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "artifacts" / "model.joblib"
    try:
        return joblib.load(model_path)
    except FileNotFoundError as exc:
        raise ValueError("Model artifact not found. Please run training first.") from exc


def load_threshold():
    project_root = Path(__file__).resolve().parents[1]
    threshold_path = project_root / "artifacts" / "threshold.json"
    try:
        with threshold_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data["selected_threshold"])
    except FileNotFoundError:
        return 0.5


def _is_valid_type(value, expected_types):
    if isinstance(value, bool):
        return False
    return isinstance(value, expected_types)


def validate_input(input_dict):
    if not isinstance(input_dict, dict):
        raise ValueError("Input must be a dictionary.")

    missing = [k for k in REQUIRED_FIELDS if k not in input_dict]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    extra = [k for k in input_dict if k not in REQUIRED_FIELDS]
    if extra:
        raise ValueError(f"Unexpected fields: {extra}")

    for field, types in REQUIRED_FIELDS.items():
        value = input_dict[field]
        if not _is_valid_type(value, types):
            raise ValueError(f"Field '{field}' must be of type {types}.")

    if input_dict["age"] <= 0:
        raise ValueError("Field 'age' must be > 0.")
    if input_dict["trestbps"] <= 0:
        raise ValueError("Field 'trestbps' must be > 0.")
    if input_dict["chol"] <= 0:
        raise ValueError("Field 'chol' must be > 0.")
    if input_dict["thalach"] <= 0:
        raise ValueError("Field 'thalach' must be > 0.")
    if input_dict["oldpeak"] < 0:
        raise ValueError("Field 'oldpeak' must be >= 0.")
    if input_dict["ca"] < 0:
        raise ValueError("Field 'ca' must be >= 0.")


def predict(input_dict):
    validate_input(input_dict)
    try:
        model = load_model()
        selected_threshold = load_threshold()
        X = pd.DataFrame([input_dict])
        prob = float(model.predict_proba(X)[0, 1])
        pred = int(prob >= selected_threshold)
        return {
            "prediction": pred,
            "probability": prob,
            "threshold_used": selected_threshold,
            "success": True,
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "success": False,
        }


if __name__ == "__main__":
    sample = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }
    print(predict(sample))
