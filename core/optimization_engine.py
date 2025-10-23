# core/optimization_engine.py
import time
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from models.registry import MODEL_REGISTRY
from core.parallel_executor import ParallelExecutor
from algorithms.genetic import GeneticOptimizer
from algorithms.pso import PSOOptimizer
from algorithms.bayesian import BayesianOptimizer
from algorithms.simulated_annealing import SimulatedAnnealingOptimizer
from algorithms.tpe import TPEOptimizer
from algorithms.random_search import RandomSearchOptimizer

class OptimizationEngine:
    def __init__(self, model_key: str, optimizer_key: str = "genetic", dataset=None, metric="accuracy", strategy_params=None):
        self.model_key = model_key
        self.optimizer_key = optimizer_key.lower()
        self.dataset = dataset
        self.metric = metric
        self.custom_metric_fn = None  # user-supplied callable
        self.strategy_params = strategy_params or {}

        cfg = MODEL_REGISTRY[model_key]
        self.search_space = cfg.build_search_space()
        self.wrapper = cfg.get_wrapper()
        self.executor = ParallelExecutor()

        # Pass optimizer-specific parameters using **self.strategy_params
        if self.optimizer_key == "pso":
            self.optimizer = PSOOptimizer(self.search_space, **self.strategy_params)
        elif self.optimizer_key == "bayesian":
            self.optimizer = BayesianOptimizer(self.search_space, **self.strategy_params)
        elif self.optimizer_key == "genetic":
            self.optimizer = GeneticOptimizer(self.search_space, **self.strategy_params)
        elif self.optimizer_key == "simulated_annealing":
            self.optimizer = SimulatedAnnealingOptimizer(self.search_space, **self.strategy_params)
        elif self.optimizer_key == "tpe":
            self.optimizer = TPEOptimizer(self.search_space, **self.strategy_params)
        elif self.optimizer_key == "random_search":
            self.optimizer = RandomSearchOptimizer(self.search_space, **self.strategy_params)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_key}")


    # ---------------- Metric Handling ---------------- #
    def set_custom_metric(self, metric_fn):
        """Allow user to provide their own custom metric function."""
        self.custom_metric_fn = metric_fn

    def _evaluate_metric(self, y_true, y_pred):
        """
        Compute chosen metric. Supports classic metrics and custom callable. Validates compatibility.
        """
        metric = self.custom_metric_fn if callable(self.custom_metric_fn) else self.metric
        if callable(metric):
            try:
                return metric(y_true, y_pred)
            except Exception as e:
                print(f"[Engine] Custom metric error: {e}")
                return float('-inf')
        m = str(metric).lower()
        try:
            if m == "accuracy":
                from sklearn.metrics import accuracy_score
                return accuracy_score(y_true, y_pred)
            elif m == "f1":
                from sklearn.metrics import f1_score
                return f1_score(y_true, y_pred, average="weighted")
            elif m == "precision":
                from sklearn.metrics import precision_score
                return precision_score(y_true, y_pred, average="weighted")
            elif m == "recall":
                from sklearn.metrics import recall_score
                return recall_score(y_true, y_pred, average="weighted")
            elif m == "roc_auc":
                from sklearn.metrics import roc_auc_score
                if hasattr(y_pred, "shape") and len(y_pred.shape) > 1:
                    return roc_auc_score(y_true, y_pred, multi_class="ovr")
                else:
                    return roc_auc_score(y_true, y_pred)
            elif m == "log_loss":
                from sklearn.metrics import log_loss
                return log_loss(y_true, y_pred)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}. Supported: accuracy, f1, precision, recall, roc_auc, log_loss or a custom callable.")
        except Exception as e:
            print(f"[Engine] Metric evaluation error: {e}")
            return float('-inf')

    # ---------------- Main Optimization ---------------- #
    def run(self, max_iters=10):
        X, y = self.dataset
        best = None
        total_start = time.time()

        print(f"[INFO] Starting optimization for model: {self.model_key}")
        print(f"[Engine] Running {self.optimizer_key.upper()} optimization for {self.model_key} (metric={self.metric})")
        metric = self.custom_metric_fn if callable(self.custom_metric_fn) else self.metric
        if callable(metric):
            print("[Engine] Using custom metric function.")
        else:
            print(f"[Engine] Using built-in metric: {str(metric).lower()}")

        for it in range(max_iters):
            iter_start = time.time()
            candidates = self.optimizer.suggest(None)
            results = self.executor.evaluate(candidates, self.wrapper, X, y)
            self.optimizer.update(results)

            for c in results:
                try:
                    preds = c.model.predict(X)
                    score = self._evaluate_metric(y, preds)
                    c.score = score
                except Exception as e:
                    print(f"[WARN] Evaluation failed: {e}")
                    c.score = float("-inf")
                if best is None or c.score > best.score:
                    best = c

            iter_time = time.time() - iter_start
            best_score = best.score if best is not None else float('-inf')
            print(f"[Engine] Iter {it+1}/{max_iters} | Best={best_score:.4f} | Time={iter_time:.2f}s")

        total_time = time.time() - total_start
        print(f"[Engine] Optimization finished in {total_time:.2f}s")
        if best is not None:
            print(f"[DONE] {self.model_key} best_score={best.score:.4f}")
            final_params = best.params
            final_model = self.wrapper.model_class(**final_params)
            final_model.fit(X, y)
            return final_model, final_params, best.score
        else:
            print(f"[DONE] {self.model_key} best_score=None")
            return None, None, None
