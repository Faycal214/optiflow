from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from typing import Callable, Optional, Tuple

class ModelWrapper:
    def __init__(self, model_class, preprocess: Optional[Callable] = None):
        """
        model_class: a scikit-learn estimator class (not instance).
        preprocess: optional function(params) -> params used to transform hyperparameters
                    before passing them to the estimator.
        """
        self.model_class = model_class
        self.preprocess = preprocess

    def train_and_score(self, params: dict, X, y, cv: int = 3, scoring="accuracy") -> float:
        """
        Evaluate params via cross-validation and return the scalar score (higher is better).
        This function does NOT return the trained estimator. It clones a fresh estimator per fold.
        """
        if self.preprocess:
            params = self.preprocess(params)

        model = self.model_class(**params)
        scores = cross_val_score(clone(model), X, y, cv=cv, scoring=scoring)
        # return mean score (higher better). Optimizers in your framework compare/scoring accordingly.
        return float(scores.mean())

    def fit_final(self, params: dict, X, y):
        """
        Instantiate and fit a final estimator on the full dataset. Returns the fitted estimator.
        Call this after selecting best params.
        """
        if self.preprocess:
            params = self.preprocess(params)
        model = self.model_class(**params)
        model.fit(X, y)
        return model
