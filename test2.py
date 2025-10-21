from core.model_search_manager import ModelSearchManager
from sklearn.datasets import load_iris, load_breast_cancer

# Custom metric example (macro F1)
def custom_macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro")

X, y = load_breast_cancer(return_X_y=True)

manager = ModelSearchManager(
    scoring="accuracy",            # or "accuracy", "rmse"
    strategy="tpe",        # or "genetic", "pso"
    custom_metric_fn=None    # or custom_macro_f1
)

# manager.search_all((X, y), max_iters=5)
manager.search_model("decision_tree", (X, y), max_iters=10)