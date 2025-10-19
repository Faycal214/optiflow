from sklearn.datasets import load_iris
from core.model_search_manager import ModelSearchManager

X, y = load_iris(return_X_y=True)

# random search
mgr = ModelSearchManager(strategy="random", n_samples=5, n_jobs=-1, cv= 2)
best_knn = mgr.search_model("knn", X, y)

# genetic search
mgr_gen = ModelSearchManager(strategy="genetic", n_samples=5,
                             strategy_params={"population": 30, "elite_frac": 0.2}, n_jobs=-1, cv= 2)
best_rf = mgr_gen.search_model("random_forest", X, y)
