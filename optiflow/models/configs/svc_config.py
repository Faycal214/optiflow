from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from optiflow.core.search_space import SearchSpace
from optiflow.core.model_wrapper import ModelWrapper


# Define this function at module level, not nested
def build_svc_pipeline(**params):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(**params))
    ])


class SVCConfig:
    name = "svc"

    @staticmethod
    def build_search_space():
        s = SearchSpace()
        s.add("C", "continuous", [1e-3, 1e3], log=True)
        s.add("kernel", "categorical", ["linear", "rbf", "poly", "sigmoid"])
        s.add("gamma", "continuous", [1e-4, 1], log=True)
        s.add("degree", "discrete", [2, 5])
        return s

    @staticmethod
    def get_wrapper():
        return ModelWrapper(build_svc_pipeline)
