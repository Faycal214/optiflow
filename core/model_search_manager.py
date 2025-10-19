# core/model_search_manager.py
import importlib
import inspect
import pkgutil
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from algorithms.random_search import RandomSearchOptimizer
from algorithms.grid_search import GridSearchOptimizer
from algorithms.genetic import GeneticOptimizer

class ModelSearchManager:
    def __init__(self, models_package="models", strategy="random", n_samples=50,
                 scoring="accuracy", cv=3, n_jobs=-1, strategy_params=None):
        self.models_package = models_package
        self.strategy = strategy
        self.n_samples = n_samples
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = None if n_jobs == -1 else n_jobs
        self.strategy_params = strategy_params or {}
        self.model_configs = self._load_all_configs()

    def _load_all_configs(self):
        package = importlib.import_module(self.models_package)
        configs = {}
        for _, modname, _ in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{self.models_package}.{modname}")
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if hasattr(obj, "name") and hasattr(obj, "build_search_space"):
                    configs[obj.name] = obj
        return configs

    def _make_optimizer(self, search_space):
        if self.strategy == "random":
            return RandomSearchOptimizer(search_space, n_samples=self.n_samples, **self.strategy_params)
        if self.strategy == "grid":
            return GridSearchOptimizer(search_space, **self.strategy_params)
        if self.strategy == "genetic":
            return GeneticOptimizer(search_space, population=self.strategy_params.get("population", 20),
                                    elite_frac=self.strategy_params.get("elite_frac", 0.2),
                                    crossover_prob=self.strategy_params.get("crossover_prob", 0.8),
                                    mutation_prob=self.strategy_params.get("mutation_prob", 0.2),
                                    seed=self.strategy_params.get("seed", None))
        raise ValueError(f"Unknown strategy: {self.strategy}")

    # worker used in ProcessPoolExecutor
    @staticmethod
    def _eval_worker(candidate, wrapper, cv, scoring, X, y):
        score = wrapper.train_and_score(candidate.params, X, y, cv=cv, scoring=scoring)
        candidate.score = score
        return candidate

    def search_model(self, model_name, X, y):
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")

        Config = self.model_configs[model_name]
        space = Config.build_search_space()
        wrapper = Config.get_wrapper()
        optimizer = self._make_optimizer(space)

        # initial suggestion and iterative loop (for population-based optimizers)
        best = None
        # For optimizers like Random/Grid return finite stream via suggest until exhausted.
        iter_count = 0
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            while True:
                candidates = optimizer.suggest(self.n_samples)
                if not candidates:
                    break
                # launch parallel evaluations
                futures = [executor.submit(self._eval_worker, c, wrapper, self.cv, self.scoring, X, y) for c in candidates]
                results = [f.result() for f in futures]
                optimizer.update(results)
                for c in results:
                    if best is None or c.score < best.score:
                        best = c
                iter_count += 1
                # small safety for non-iterative strategies
                if self.strategy in ("random", "grid"):
                    break
        return best

    def search_all(self, X, y):
        summary = {}
        for name in self.model_configs:
            print(f"Searching {name} using {self.strategy}")
            best = self.search_model(name, X, y)
            summary[name] = best
        return summary
