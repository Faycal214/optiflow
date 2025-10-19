from sklearn.model_selection import cross_val_score

class ModelWrapper:
    def __init__(self, model_class, preprocess=None):
        self.model_class = model_class
        self.preprocess = preprocess  # optional preprocessing hook

    def train_and_score(self, params, X, y, cv=3, scoring="accuracy"):
        # Apply preprocessing if defined
        if self.preprocess is not None:
            params = self.preprocess(params)

        model = self.model_class(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return scores.mean()
