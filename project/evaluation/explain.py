import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


def _load_data(data_path: Path):
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def _resolve_base_estimator(model_step):
    if hasattr(model_step, "calibrated_classifiers_") and model_step.calibrated_classifiers_:
        return model_step.calibrated_classifiers_[0].estimator
    if hasattr(model_step, "estimator"):
        return model_step.estimator
    return model_step


def _get_feature_names(preprocess, input_features):
    try:
        return list(preprocess.get_feature_names_out())
    except Exception:
        names = []
        for _, transformer, cols in preprocess.transformers_:
            if transformer == "drop":
                continue
            if cols is None:
                cols = input_features
            if isinstance(cols, slice):
                cols = input_features[cols]
            if isinstance(cols, (np.ndarray, pd.Index)):
                cols = list(cols)
            if transformer == "passthrough":
                names.extend(list(cols))
                continue
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    out = transformer.get_feature_names_out(cols)
                except TypeError:
                    out = transformer.get_feature_names_out()
                names.extend(list(out))
            else:
                names.extend(list(cols))
        return names


def _save_importances(path: Path, feature_names, importances, extra_cols=None):
    data = {
        "feature": feature_names,
        "importance": importances,
    }
    if extra_cols:
        data.update(extra_cols)
    df = pd.DataFrame(data)
    df = df.sort_values("importance", ascending=False)
    df.to_csv(path, index=False)


def _plot_bar(path: Path, feature_names, importances, title):
    order = np.argsort(importances)[::-1]
    top_k = min(20, len(order))
    idx = order[:top_k][::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_k), np.array(importances)[idx])
    plt.yticks(range(top_k), np.array(feature_names)[idx])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    warnings.filterwarnings("ignore")

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "heart.csv"
    model_path = project_root / "artifacts" / "model.joblib"
    out_dir = project_root / "artifacts" / "explainability"
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = joblib.load(model_path)
    X, y = _load_data(data_path)

    model_step = pipeline.named_steps.get("model", pipeline)
    base_estimator = _resolve_base_estimator(model_step)
    model_type = base_estimator.__class__.__name__

    explanation_method = "permutation_importance"
    generated_files = []

    shap_summary_path = out_dir / "shap_summary_bar.png"
    global_importance_path = out_dir / "global_feature_importance.csv"

    shap_used = False
    try:
        import shap

        features_step = pipeline.named_steps["features"]
        preprocess_step = pipeline.named_steps["preprocess"]
        X_fe = features_step.transform(X)
        X_pre = preprocess_step.transform(X_fe)

        feature_names = _get_feature_names(preprocess_step, list(X_fe.columns))

        if hasattr(base_estimator, "coef_"):
            rng = np.random.RandomState(42)
            sample_size = min(500, X_pre.shape[0])
            sample_idx = rng.choice(X_pre.shape[0], size=sample_size, replace=False)
            X_sample = X_pre[sample_idx]

            explainer = shap.LinearExplainer(base_estimator, X_sample)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[-1]

            mean_abs = np.mean(np.abs(shap_values), axis=0)
            _save_importances(global_importance_path, feature_names, mean_abs)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
            )
            plt.tight_layout()
            plt.savefig(shap_summary_path, dpi=200)
            plt.close()

            explanation_method = "shap_linear_explainer"
            shap_used = True
    except Exception:
        shap_used = False

    if not shap_used:
        perm = permutation_importance(
            pipeline,
            X,
            y,
            n_repeats=10,
            random_state=42,
            scoring="roc_auc",
        )
        feature_names = list(X.columns)
        _save_importances(
            global_importance_path,
            feature_names,
            perm.importances_mean,
            extra_cols={"importance_std": perm.importances_std},
        )
        _plot_bar(
            shap_summary_path,
            feature_names,
            perm.importances_mean,
            "Permutation Importance (Global)",
        )
        explanation_method = "permutation_importance"

    generated_files.extend(
        [
            shap_summary_path.name,
            global_importance_path.name,
        ]
    )

    metadata = {
        "model_type": model_type,
        "explanation_method": explanation_method,
        "generated_files": generated_files,
    }
    with (out_dir / "explainability_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
