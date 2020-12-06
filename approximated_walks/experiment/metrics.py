from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score
from sanitize_ml_labels import sanitize_ml_labels


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return metrics computed for given predictions.

    Parameters
    -----------------
    y_true: np.ndarray,
        Ground truth labels.
    y_pred: np.ndarray,
        Model predictions

    Returns
    -----------------
    Dictionary with predictions.
    """
    integer_metrics = (
        accuracy_score,
        balanced_accuracy_score
    )
    float_metrics = (
        roc_auc_score,
        average_precision_score
    )
    return {
        "F1 Score Binary": f1_score(y_true, np.round(y_pred), average="binary"),
        "F1 Score Micro": f1_score(y_true, np.round(y_pred), average="micro"),
        "F1 Score Macro": f1_score(y_true, np.round(y_pred), average="macro"),
        **{
            sanitize_ml_labels(metric.__name__): metric(y_true, np.round(y_pred))
            for metric in integer_metrics
        },
        **{
            sanitize_ml_labels(metric.__name__): metric(y_true, y_pred)
            for metric in float_metrics
        }
    }
