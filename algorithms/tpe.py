import optuna
from core.base_optimizer import BaseOptimizer, Candidate

class TPEOptimizer(BaseOptimizer):
    def __init__(self, search_space, metric="accuracy", population_size=10):
        super().__init__()
        self.search_space = search_space
        self.metric = metric
        self.population_size = population_size
        self.study = optuna.create_study(direction="maximize")
        self.trials = []

    def _define_search_space(self, trial):
        params = {}
        for name, cfg in self.search_space.parameters.items():   # fixed here
            if isinstance(cfg, list) and len(cfg) == 2 and all(isinstance(v, (int, float)) for v in cfg):
                params[name] = trial.suggest_float(name, cfg[0], cfg[1])
            elif isinstance(cfg, list) and all(isinstance(v, str) for v in cfg):
                params[name] = trial.suggest_categorical(name, cfg)
            elif isinstance(cfg, tuple) and len(cfg) == 2:
                params[name] = trial.suggest_float(name, cfg[0], cfg[1])
            else:
                raise ValueError(f"Unsupported search space format for {name}: {cfg}")
        return params

    def suggest(self, history=None):
        candidates = []
        for _ in range(self.population_size):
            trial = self.study.ask()
            params = self._define_search_space(trial)
            candidates.append(Candidate(params=params))
        return candidates

    def update(self, results):
        for candidate, score in results:
            self.study.tell(candidate.params, score)
            self.trials.append((candidate.params, score))

    def get_best(self):
        if not self.trials:
            return None, None
        best_params, best_score = max(self.trials, key=lambda x: x[1])
        return best_params, best_score
