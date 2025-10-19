# core/optimize_pipeline.py
from core.model_search_manager import ModelSearchManager
from sklearn.datasets import load_iris

if __name__ == "__main__":
    X, y = load_iris(return_X_y=True)
    manager = ModelSearchManager(strategy="genetic", scoring="accuracy")
    results = manager.search_all((X, y), max_iters=5)
