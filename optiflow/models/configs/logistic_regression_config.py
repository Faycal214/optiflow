from sklearn.linear_model import LogisticRegression
from optiflow.core.search_space import SearchSpace
from optiflow.core.model_wrapper import ModelWrapper

class LogisticRegressionConfig:
    name = "logistic_regression"

    @staticmethod
    def build_search_space():
        s = SearchSpace()
        s.add("C", "continuous", [1e-5, 1e4], log=True)
        s.add("penalty", "categorical", ["l1", "l2", "elasticnet", "none"])
        s.add("solver", "categorical", ["liblinear", "lbfgs", "newton-cg", "saga"])
        s.add("max_iter", "discrete", [100, 10000])
        s.add("fit_intercept", "categorical", [True, False])
        s.add("class_weight", "categorical", [None, "balanced"])
        s.add("l1_ratio", "continuous", [0.0, 1.0])
        s.add("tol", "continuous", [1e-6, 1e-2], log=True)
        return s

    @staticmethod
    def get_wrapper():
        return ModelWrapper(LogisticRegression)
