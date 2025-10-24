from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Callable
import inspect
import numpy as np

METRICS = {
    "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro"),
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro"),
}

def get_metric(metric):
    """Return callable(y_true, y_pred)->float or raise."""
    if callable(metric):
        # quick signature check: must accept (y_true, y_pred)
        sig = inspect.signature(metric)
        if len(sig.parameters) >= 2:
            return metric
        raise ValueError("Custom metric must accept at least (y_true, y_pred).")
    if isinstance(metric, str):
        if metric in METRICS:
            return METRICS[metric]
        raise ValueError(f"Unknown metric: {metric}")
    raise TypeError("metric must be str or callable")
