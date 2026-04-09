import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.required_columns_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.required_columns_ = list(X.columns)
        else:
            self.required_columns_ = None
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.required_columns_)
        X = X.copy()

        X["low_thalach"] = (X["thalach"] < 140).astype(int)
        X["high_oldpeak"] = (X["oldpeak"] > 2).astype(int)
        X["age_group"] = pd.cut(
            X["age"],
            bins=[0, 40, 55, 70, 100],
            labels=[0, 1, 2, 3],
        ).astype(int)
        X["has_blockage"] = (X["ca"] > 0).astype(int)
        X["exercise_risk"] = (X["exang"] + X["high_oldpeak"]).astype(int)

        return X


def build_pipeline():
    continuous_features = ["age", "trestbps", "thalach", "oldpeak"]
    binary_features = [
        "sex",
        "fbs",
        "exang",
        "ca",
        "low_thalach",
        "high_oldpeak",
        "has_blockage",
        "exercise_risk",
    ]
    categorical_features = [
        "cp",
        "restecg",
        "slope",
        "thal",
        "age_group",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, continuous_features),
            ("bin", binary_transformer, binary_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    base_model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )

    model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5,
    )

    pipeline = Pipeline(
        steps=[
            ("features", FeatureEngineering()),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline
