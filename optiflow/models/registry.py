# optiflow/models/registry.py
"""
Central model registry.
It imports the config classes you already have in models/configs and exposes:
 - MODEL_REGISTRY dict
 - get_model_config(name) helper
"""

from optiflow.models.configs.svc_config import SVCConfig
from optiflow.models.configs.random_forest_config import RandomForestConfig
from optiflow.models.configs.xgboost_config import XGBoostConfig
from optiflow.models.configs.mlp_config import MLPConfig
from optiflow.models.configs.decision_tree_config import DecisionTreeConfig
from optiflow.models.configs.knn_config import KNNConfig
from optiflow.models.configs.logistic_regression_config import LogisticRegressionConfig
# add imports only for config files that actually exist

MODEL_REGISTRY = {
    "svc": SVCConfig,
    "random_forest": RandomForestConfig,
    "xgboost": XGBoostConfig,
    "mlp": MLPConfig,
    "decision_tree": DecisionTreeConfig,
    "knn": KNNConfig,
    "logistic_regression": LogisticRegressionConfig,
}

def get_model_config(name: str):
    """Return the config class for a model key (e.g., 'svc')."""
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
