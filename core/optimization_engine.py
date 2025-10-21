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
from algorithms.grid_search import GridSearchOptimizer
from algorithms.random_search import RandomSearchOptimizer

class OptimizationEngine:
    def __init__(self, model_key: str, optimizer_key: str = "genetic", dataset=None, metric="accuracy"):
        self.model_key = model_key
        self.optimizer_key = optimizer_key.lower()
        self.dataset = dataset
        self.metric = metric
        self.custom_metric_fn = None  # user-supplied callable

        cfg = MODEL_REGISTRY[model_key]
        self.search_space = cfg.build_search_space()
        self.wrapper = cfg.get_wrapper()
        self.executor = ParallelExecutor()

        if self.optimizer_key == "pso":
            self.optimizer = PSOOptimizer(self.search_space)
        elif self.optimizer_key == "bayesian":
            self.optimizer = BayesianOptimizer(self.search_space)
        elif self.optimizer_key == "genetic":
            self.optimizer = GeneticOptimizer(self.search_space)
        elif self.optimizer_key == "simulated_annealing":
            self.optimizer = SimulatedAnnealingOptimizer(self.search_space)
        elif self.optimizer_key == "tpe":
            self.optimizer = TPEOptimizer(self.search_space)
        elif self.optimizer_key == "grid_search":
            self.optimizer = GridSearchOptimizer(self.search_space)
        elif self.optimizer_key == "random_search":
            self.optimizer = RandomSearchOptimizer(self.search_space)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_key}")


    # ---------------- Metric Handling ---------------- #
    def set_custom_metric(self, metric_fn):
        """Allow user to provide their own custom metric function."""
        self.custom_metric_fn = metric_fn

    def _evaluate_metric(self, y_true, y_pred):
        """Compute chosen metric."""
        if self.custom_metric_fn:
            return self.custom_metric_fn(y_true, y_pred)
        m = self.metric.lower()
        if m == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif m == "f1":
            return f1_score(y_true, y_pred, average="weighted")
        elif m == "rmse":
            return mean_squared_error(y_true, y_pred, squared=False)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    # ---------------- Main Optimization ---------------- #
    def run(self, max_iters=10):
        X, y = self.dataset
        best = None
        total_start = time.time()

        print(f"[Engine] Running {self.optimizer_key.upper()} optimization for {self.model_key} (metric={self.metric})")
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
            print(f"[Engine] Iter {it+1}/{max_iters} | Best={best.score:.4f} | Time={iter_time:.2f}s")

        total_time = time.time() - total_start
        print(f"[Engine] Optimization finished in {total_time:.2f}s")

        final_params = best.params
        final_model = self.wrapper.model_class(**final_params)
        final_model.fit(X, y)

        return final_model, final_params, best.score
