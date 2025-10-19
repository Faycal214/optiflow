from sklearn.svm import SVC
from core.search_space import SearchSpace
from core.model_wrapper import ModelWrapper

class SVCConfig:
    name = "svc"

    @staticmethod
    def build_search_space():
        s = SearchSpace()
        s.add("C", "continuous", [1e-4, 1e4], log=True)
        s.add("kernel", "categorical", ["linear", "rbf", "poly", "sigmoid"])
        s.add("gamma", "continuous", [1e-5, 10], log=True)
        s.add("degree", "discrete", [2, 8])
        s.add("coef0", "continuous", [-1.0, 1.0])
        s.add("shrinking", "categorical", [True, False])
        s.add("tol", "continuous", [1e-5, 1e-1], log=True)
        s.add("max_iter", "discrete", [100, 10000])
        return s

    @staticmethod
    def get_wrapper():
        return ModelWrapper(SVC)
