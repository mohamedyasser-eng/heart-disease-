import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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
    non_continuous_features = [
        "sex",
        "cp",
        "chol",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal",
        "low_thalach",
        "high_oldpeak",
        "age_group",
        "has_blockage",
        "exercise_risk",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    other_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, continuous_features),
            ("other", other_transformer, non_continuous_features),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000,
    )

    pipeline = Pipeline(
        steps=[
            ("features", FeatureEngineering()),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline
