# models/registry.py
from core.search_space import SearchSpace
from core.model_wrapper import SklearnWrapper
from sklearn.svm import SVC

class SVCConfig:
    @staticmethod
    def build_search_space():
        s = SearchSpace()
        s.add("C", "categorical", [0.1, 1, 10, 100])
        s.add("gamma", "categorical", [1, 0.1, 0.01, 0.001])
        s.add("kernel", "categorical", ["rbf","linear"])
        return s

    @staticmethod
    def get_wrapper():
        return SklearnWrapper(SVC)

MODEL_REGISTRY = {
    "svc": SVCConfig
}
