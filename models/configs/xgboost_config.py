from xgboost import XGBClassifier
from core.search_space import SearchSpace
from core.model_wrapper import ModelWrapper

class XGBoostConfig:
    name = "xgboost"

    @staticmethod
    def build_search_space():
        s = SearchSpace()
        s.add("n_estimators", "discrete", [50, 3000])
        s.add("max_depth", "discrete", [2, 20])
        s.add("learning_rate", "continuous", [1e-4, 0.5], log=True)
        s.add("subsample", "continuous", [0.1, 1.0])
        s.add("colsample_bytree", "continuous", [0.1, 1.0])
        s.add("colsample_bylevel", "continuous", [0.1, 1.0])
        s.add("gamma", "continuous", [0.0, 10.0])
        s.add("min_child_weight", "discrete", [1, 50])
        s.add("reg_lambda", "continuous", [0.0, 20.0])
        s.add("reg_alpha", "continuous", [0.0, 20.0])
        s.add("booster", "categorical", ["gbtree", "gblinear", "dart"])
        s.add("scale_pos_weight", "continuous", [0.5, 10.0])
        s.add("grow_policy", "categorical", ["depthwise", "lossguide"])
        return s

    @staticmethod
    def get_wrapper():
        return ModelWrapper(XGBClassifier)
