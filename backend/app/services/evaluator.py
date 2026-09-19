"""
evaluator.py

Computes classification metrics on a trained pipeline's test-set predictions.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


def evaluate_classification(pipeline, X_test, y_test) -> dict:
    """
    y_test is expected to already be label-encoded to 0/1 (see trainer.py's
    encode_target()) -- this function stays generic and doesn't need to know
    about the original string labels ("Y"/"N", "Approved"/"Rejected", etc).
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, pos_label=1)), 4),
        "recall": round(float(recall_score(y_test, y_pred, pos_label=1)), 4),
        "f1": round(float(f1_score(y_test, y_pred, pos_label=1)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics