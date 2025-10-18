# core/optimization_engine.py
from models.registry import MODEL_REGISTRY
from core.parallel_executor import ParallelExecutor
from algorithms.genetic import GeneticOptimizer
from typing import Tuple

class OptimizationEngine:
    def __init__(self, model_key: str, optimizer_key: str = "genetic", dataset=None, metric="accuracy"):
        self.model_key = model_key
        self.optimizer_key = optimizer_key
        self.dataset = dataset
        self.metric = metric

        cfg = MODEL_REGISTRY[model_key]
        self.search_space = cfg.build_search_space()
        self.wrapper = cfg.get_wrapper()
        # only genetic for starter
        self.optimizer = GeneticOptimizer(self.search_space)
        self.executor = ParallelExecutor()

    def run(self, max_iters=10):
        X, y = self.dataset
        best = None
        for it in range(max_iters):
            candidates = self.optimizer.suggest(None)
            results = self.executor.evaluate(candidates, self.wrapper, X, y)
            self.optimizer.update(results)
            # track best
            for c in results:
                if best is None or c.score < best.score:
                    best = c
            print(f"iter {it+1} best_score {best.score:.4f}")
        # return best model trained on full data (sklearn wrapper returns negative cv score)
        # build final estimator with best params
        final_params = best.params
        # instantiate and fit final model
        final_model = self.wrapper.model_cls(**final_params)
        final_model.fit(X, y)
        return final_model, final_params, best.score
