from sklearn.model_selection import cross_val_score
from sklearn.base import clone

class ModelWrapper:
    def __init__(self, model_class, preprocess=None):
        self.model_class = model_class
        self.preprocess = preprocess

    def train_and_score(self, params, X, y, cv=3, scoring="accuracy"):
        """
        Train model with given params using cross-validation.
        """
        if self.preprocess:
            params = self.preprocess(params)

        model = self.model_class(**params)

        # clone ensures a fresh model per fold (fitted internally)
        scores = cross_val_score(clone(model), X, y, cv=cv, scoring=scoring)

        # fit final model on full data (optional)
        model.fit(X, y)

        return scores.mean(), model
