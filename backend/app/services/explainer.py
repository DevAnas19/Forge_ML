"""
explainer.py

SHAP-based explainability -- global feature importance and single
prediction explanations. Ported from Kaggle Phase 5 Steps 1-3.
"""

import numpy as np
import shap


def get_shap_explainer(model, model_name: str, X_background=None):
    """
    Tree-based models (Random Forest, XGBoost) use the fast, exact
    TreeExplainer. Logistic Regression -- a linear model -- needs
    LinearExplainer instead, which requires a background dataset to
    estimate feature distributions.
    """
    if model_name in ("random_forest", "xgboost"):
        return shap.TreeExplainer(model)
    elif model_name == "logistic_regression":
        return shap.LinearExplainer(model, X_background)
    else:
        raise ValueError(f"No SHAP explainer configured for model: {model_name}")


def _get_feature_columns(preprocessor):
    """
    Pulls the original raw column names back out of a fitted ColumnTransformer,
    rather than requiring the caller to pass them in separately -- the
    preprocessor already knows exactly which columns it was fit on.
    """
    numerical_cols = preprocessor.transformers_[0][2]
    categorical_cols = preprocessor.transformers_[1][2]
    return list(numerical_cols) + list(categorical_cols)


def _transform(preprocessor, df):
    transformed = preprocessor.transform(df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return transformed


def compute_global_importance(pipeline, model_name: str, raw_df, sample_size: int = 100) -> dict:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_cols = _get_feature_columns(preprocessor)
    sample_df = raw_df[feature_cols].sample(min(sample_size, len(raw_df)), random_state=42)

    X_transformed = _transform(preprocessor, sample_df)
    feature_names = preprocessor.get_feature_names_out()

    explainer = get_shap_explainer(model, model_name, X_background=X_transformed)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    ranked = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "feature_importance": [
            {"feature": name, "importance": round(float(value), 4)}
            for name, value in ranked
        ]
    }


def explain_prediction(pipeline, model_name: str, input_row_df, raw_df_for_background, sample_size: int = 100) -> dict:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_cols = _get_feature_columns(preprocessor)
    feature_names = preprocessor.get_feature_names_out()

    background_df = raw_df_for_background[feature_cols].sample(
        min(sample_size, len(raw_df_for_background)), random_state=42
    )
    X_background = _transform(preprocessor, background_df)

    transformed_row = _transform(preprocessor, input_row_df[feature_cols])

    explainer = get_shap_explainer(model, model_name, X_background=X_background)
    shap_values = explainer.shap_values(transformed_row)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    contributions = shap_values[0]

    explanation = sorted(
        zip(feature_names, contributions),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return {
        "feature_contributions": [
            {"feature": name, "contribution": round(float(value), 4)}
            for name, value in explanation
        ]
    }