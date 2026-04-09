from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_model():
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "artifacts" / "model.joblib"
    return joblib.load(model_path)


def predict(input_dict):
    model = load_model()
    X = pd.DataFrame([input_dict])
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0, 1])
    return {
        "prediction": pred,
        "probability": prob,
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
