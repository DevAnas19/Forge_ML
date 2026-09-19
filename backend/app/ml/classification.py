"""
classification.py

Train/test split logic, plus a registry mapping model names to sklearn
estimator classes -- this is what makes models "pluggable" (spec section 29,
rule 9): trainer.py never needs an if/elif chain per model type, it just
looks the requested name up here.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Add a new classification model here and it becomes available everywhere
# (API, trainer) with no other code changes.
CLASSIFICATION_MODELS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(random_state=42),
    "xgboost": lambda: XGBClassifier(random_state=42, eval_metric="logloss"),
}


def get_classification_model(model_name: str):
    if model_name not in CLASSIFICATION_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(CLASSIFICATION_MODELS.keys())}"
        )
    return CLASSIFICATION_MODELS[model_name]()


def split_data(df, target_column: str, feature_cols: list, test_size: float = 0.2, random_seed: int = 42):
    X = df[feature_cols]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )

    return X_train, X_test, y_train, y_test