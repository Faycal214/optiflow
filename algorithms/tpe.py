import optuna
from core.base_optimizer import BaseOptimizer, Candidate


from typing import Any, Dict, List, Optional, Tuple

class TPEOptimizer(BaseOptimizer):
    """
    Tree-structured Parzen Estimator (TPE) optimizer using Optuna.
    Handles int, float, and categorical parameters from SearchSpace.
    """
    def __init__(self, search_space, metric: str = "accuracy", population_size: int = 10):
        super().__init__()
        self.search_space = search_space
        self.metric = metric
        self.population_size = population_size
        self.study = optuna.create_study(direction="maximize")
        self.trials: List[Tuple[Dict[str, Any], float]] = []

    def _define_search_space(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        """
        Suggest parameters for a trial based on SearchSpace definition.
        """
        params = {}
        for name, cfg in self.search_space.parameters.items():
            ptype = cfg["type"]
            values = cfg["values"]
            log = cfg.get("log", False)
            if ptype == "int" or ptype == "discrete":
                # Discrete: range or list
                if isinstance(values, (list, tuple)) and len(values) == 2 and all(isinstance(x, int) for x in values):
                    params[name] = trial.suggest_int(name, values[0], values[1], log=log)
                else:
                    params[name] = trial.suggest_categorical(name, list(values))
            elif ptype == "float" or ptype == "continuous":
                params[name] = trial.suggest_float(name, values[0], values[1], log=log)
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, list(values))
            else:
                raise ValueError(f"[TPE] Unsupported parameter type: {ptype} for {name}")
        return params

    def suggest(self, history: Optional[List[Candidate]] = None) -> List[Candidate]:
        """
        Suggest a batch of candidates by sampling parameters from Optuna.
        """
        print("[TPE] Suggesting new parameters...")
        candidates = []
        for _ in range(self.population_size):
            trial = self.study.ask()
            params = self._define_search_space(trial)
            candidate = Candidate(params=params)
            candidate.trial = trial  # Attach trial object for later use
            candidates.append(candidate)
        return candidates

    def update(self, results: List[Candidate]) -> None:
        """
        Update Optuna study with evaluated candidate results.
        """
        for i, candidate in enumerate(results, 1):
            score = candidate.score
            trial = getattr(candidate, "trial", None)
            if trial is not None:
                self.study.tell(trial, score)
            else:
                print(f"[TPE] Warning: Candidate missing trial object, skipping Optuna tell.")
            self.trials.append((candidate.params, score))
            print(f"[TPE] Completed iteration {i}/{len(results)} with score={score}")

    def get_best(self) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[float]]:
        """
        Return best parameters and score found so far.
        """
        if not self.trials:
            return None, None, None
        best_params, best_score = max(self.trials, key=lambda x: x[1])
        # Model instantiation is handled by OptimizationEngine
        return None, best_params, best_score
