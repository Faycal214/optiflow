# core/pipeline.py
from optiflow.core.optimization_engine import OptimizationEngine

class MLPipeline:
    def __init__(self, model_key, dataset, optimizer="genetic", metric="accuracy"):
        self.model_key = model_key
        self.dataset = dataset
        self.optimizer = optimizer
        self.metric = metric

    def train(self, max_iters=5):
        print(f"[Pipeline] Starting optimization for model: {self.model_key}")

        engine = OptimizationEngine(
            model_key=self.model_key,
            optimizer_key=self.optimizer,
            dataset=self.dataset,
            metric=self.metric,
        )

        best_model, best_params, best_score = engine.run(max_iters=max_iters)

        print(f"[Pipeline] Optimization complete.")
        print(f"[Pipeline] Best score: {best_score:.4f}")
        print(f"[Pipeline] Best parameters: {best_params}")

        return best_model, best_params, best_score
