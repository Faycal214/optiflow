# core/optimization_engine.py
import time
from models.registry import MODEL_REGISTRY
from core.parallel_executor import ParallelExecutor
from algorithms.genetic import GeneticOptimizer
from algorithms.pso import PSOOptimizer

class OptimizationEngine:
    def __init__(self, model_key: str, optimizer_key: str = "genetic", dataset=None, metric="accuracy"):
        self.model_key = model_key
        self.optimizer_key = optimizer_key.lower()
        self.dataset = dataset
        self.metric = metric

        cfg = MODEL_REGISTRY[model_key]
        self.search_space = cfg.build_search_space()
        self.wrapper = cfg.get_wrapper()
        self.executor = ParallelExecutor()

        if self.optimizer_key == "pso":
            self.optimizer = PSOOptimizer(self.search_space)
        elif self.optimizer_key == "genetic":
            self.optimizer = GeneticOptimizer(self.search_space)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_key}")

    def run(self, max_iters=10):
        X, y = self.dataset
        best = None
        total_start = time.time()

        print(f"[Engine] Running {self.optimizer_key.upper()} optimization for {self.model_key}")
        for it in range(max_iters):
            iter_start = time.time()

            candidates = self.optimizer.suggest(None)
            results = self.executor.evaluate(candidates, self.wrapper, X, y)
            self.optimizer.update(results)

            for c in results:
                if best is None or c.score < best.score:
                    best = c

            iter_time = time.time() - iter_start
            print(f"[Engine] Iter {it+1}/{max_iters} | Best={best.score:.4f} | Time={iter_time:.2f}s")

        total_time = time.time() - total_start
        print(f"[Engine] Optimization finished in {total_time:.2f}s")

        final_params = best.params
        final_model = self.wrapper.model_class(**final_params)
        final_model.fit(X, y)

        return final_model, final_params, best.score
