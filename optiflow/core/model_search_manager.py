# core/model_search_manager.py
import importlib
import inspect
import pkgutil
from typing import Dict, Any, Tuple
from optiflow.models.registry import MODEL_REGISTRY
from optiflow.core.optimization_engine import OptimizationEngine


class ModelSearchManager:
    def __init__(self,
                 models_package: str = "models",
                 strategy: str = "genetic",
                 n_samples: int = 50,
                 scoring: str = "accuracy",
                 cv: int = 3,
                 n_jobs: int = -1,
                 strategy_params: dict = None,
                 custom_metric_fn=None):
        self.models_package = models_package
        self.strategy = strategy
        self.n_samples = n_samples
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = None if n_jobs == -1 else n_jobs
        self.strategy_params = strategy_params or {}
        self.custom_metric_fn = custom_metric_fn
        self.model_configs = self._load_all_configs()

    # ---------------- Auto-load all models ---------------- #
    def _load_all_configs(self) -> Dict[str, Any]:
        """Auto-load all model configs from the given package."""
        package = importlib.import_module(self.models_package)
        configs = {}
        for _, modname, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{self.models_package}.{modname}")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if hasattr(obj, "name") and hasattr(obj, "build_search_space"):
                    configs[obj.name] = obj
        return configs

    # ---------------- Run for single model ---------------- #
    def search_model(self, model_name: str, dataset: Tuple, max_iters: int = 10):
        """Run optimization for a single model using OptimizationEngine."""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")

        print(f"\n[INFO] Starting optimization for model: {model_name}")
        engine = OptimizationEngine(
            model_key=model_name,
            optimizer_key=self.strategy,
            dataset=dataset,
            metric=self.scoring,
            strategy_params=self.strategy_params
        )

        # Optional custom metric
        if self.custom_metric_fn:
            engine.set_custom_metric(self.custom_metric_fn)

        model, params, score = engine.run(max_iters=max_iters)
        if score is not None:
            print(f"[DONE] {model_name} best_score={score:.4f}")
        else:
            print(f"[DONE] {model_name} best_score=None")
        return {"model": model, "params": params, "score": score}

    # ---------------- Run for all registered models ---------------- #
    def search_all(self, dataset: Tuple, max_iters: int = 10):
        """Run optimization for all registered models."""
        results = []
        for model_name in MODEL_REGISTRY.keys():
            res = self.search_model(model_name, dataset, max_iters)
            res["model_name"] = model_name
            results.append(res)

        # Sort models by descending score
        results = sorted(results, key=lambda r: r["score"], reverse=True)

        print("\n[SUMMARY] Best models:")
        for rank, r in enumerate(results, 1):
            print(f"{rank}. {r['model_name']} -> score={r['score']:.4f}")
        return results
