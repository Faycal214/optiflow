# core/model_wrapper.py
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
import numpy as np

class SklearnWrapper:
    def __init__(self, model_cls):
        # model_cls is the class (e.g., sklearn.svm.SVC)
        self.model_cls = model_cls

    def train_and_score(self, params: dict, X, y, cv=3, scoring="accuracy", random_state=42):
        model = self.model_cls(**params)
        # cross_val_score returns array; higher is better
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        # we return a scalar cost (lower is better) to standardize
        return -float(np.mean(scores))  # negative accuracy as cost
