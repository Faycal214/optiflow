from xgboost import XGBClassifier
from optiflow.core.search_space import SearchSpace
from optiflow.core.model_wrapper import ModelWrapper

class XGBoostConfig:
    name = "xgboost"

    @staticmethod
    def build_search_space():
        s = SearchSpace()
        # Core structure
        s.add("n_estimators", "discrete", [50, 100, 200])
        s.add("max_depth", "discrete", [3, 5, 7])
        s.add("learning_rate", "continuous", [1e-3, 0.3], log=True)

        # Sampling + regularization
        s.add("subsample", "continuous", [0.5, 1.0])
        s.add("colsample_bytree", "continuous", [0.5, 1.0])
        s.add("gamma", "continuous", [0.0, 5.0])
        s.add("min_child_weight", "discrete", [1, 10])
        s.add("reg_lambda", "continuous", [0.0, 10.0])
        s.add("reg_alpha", "continuous", [0.0, 10.0])
        return s

    @staticmethod
    def get_wrapper():
        return ModelWrapper(XGBClassifier)
