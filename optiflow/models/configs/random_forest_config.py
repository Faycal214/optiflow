from sklearn.ensemble import RandomForestClassifier
from optiflow.core.search_space import SearchSpace
from optiflow.core.model_wrapper import ModelWrapper
import random

class RandomForestConfig:
    name = "random_forest"

    @staticmethod
    def build_search_space():
        s = SearchSpace()
        s.add("n_estimators", "discrete", [20, 100])
        s.add("max_depth", "discrete", [2, 20])
        s.add("min_samples_split", "discrete", [2, 10])
        s.add("min_samples_leaf", "discrete", [1, 5])
        s.add("bootstrap", "categorical", [True, False])
        s.add("max_features", "categorical", ["sqrt", "log2", None])
        return s

    @staticmethod
    def get_wrapper():
        return ModelWrapper(RandomForestClassifier, preprocess=RandomForestConfig._preprocess_params)

    @staticmethod
    def _preprocess_params(params):
        # fix invalid combinations
        if not params["bootstrap"]:
            params["max_samples"] = None
        else:
            # sample only when bootstrap=True
            params["max_samples"] = random.uniform(0.3, 1.0)
        return params
