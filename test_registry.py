# tools/test_registry.py
from optiflow.models import get_model_config

def test():
    for key in ["svc", "random_forest", "mlp", "decision_tree", "knn", "logistic_regression", "xgboost"]:
        try:
            cfg = get_model_config(key)
            space = cfg.build_search_space()
            wrapper = cfg.get_wrapper()
            print(f"[OK] {key}: search_space keys = {list(space.parameters.keys())} wrapper = {type(wrapper)}")
        except Exception as e:
            print(f"[ERR] {key}: {e}")

if __name__ == "__main__":
    test()
