# algorithms/bayesian.py
import random
from typing import List
from core.base_optimizer import BaseOptimizer, Candidate
from core.search_space import SearchSpace
from skopt import Optimizer as SkOptimizer
from skopt.space import Real, Integer, Categorical


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, search_space: SearchSpace, n_initial_points: int = 5, random_state: int = None):
        self.space = search_space
        self.n_initial_points = n_initial_points
        self.random_state = random_state or 42

        # convert search space to skopt format
        self.sk_space = self._to_skopt_space()
        self.optimizer = SkOptimizer(
            dimensions=self.sk_space,
            random_state=self.random_state,
            n_initial_points=self.n_initial_points
        )

        # store evaluated points
        self.trials = []  # <— missing before
        self.results = []  # optional tracking

    def suggest(self, n: int = None) -> List[Candidate]:
        # initial samples if no prior evaluations
        if len(self.trials) == 0:
            samples = [self.space.sample() for _ in range(self.n_initial_points)]
            return [Candidate(s) for s in samples]

        # Bayesian suggestion from skopt
        suggestions = self.optimizer.ask(n_points=n or 1)
        param_names = list(self.space.parameters.keys())
        candidates = []
        for s in suggestions:
            cand_params = dict(zip(param_names, s))
            candidates.append(Candidate(cand_params))
        return candidates

    def update(self, results: List[Candidate]):
        for cand in results:
            x = [cand.params[k] for k in self.space.parameters.keys()]
            y = -cand.score

            # --- Sanitize parameters to ensure they are in space bounds ---
            x_clean = []
            for xi, (name, info) in zip(x, self.space.parameters.items()):
                t = info["type"]
                v = info["values"]

                if t in ["continuous", "int"]:
                    low, high = v
                    # clamp numeric values into range
                    xi = max(min(xi, high), low)
                elif t in ["categorical", "discrete"]:
                    if xi not in v:
                        xi = random.choice(v)
                x_clean.append(xi)
            # -------------------------------------------------------------

            self.trials.append((x_clean, y))
            self.optimizer.tell(x_clean, y)
        self.results.extend(results)


    def _to_skopt_space(self):
        out = []
        for name, info in self.space.parameters.items():
            t = info["type"]
            v = info["values"]
            log = info.get("log", False)

            if t == "continuous":
                low, high = v
                if log and (low <= 0 or high <= 0):
                    log = False
                out.append(Real(low, high, prior="log-uniform" if log else "uniform", name=name))

            elif t == "int":
                low, high = v
                out.append(Integer(low, high, name=name))

            elif t == "discrete":
                out.append(Categorical(v, name=name))

            elif t == "categorical":
                out.append(Categorical(v, name=name))

            else:
                raise ValueError(f"Unknown parameter type {t} for {name}")
        return out
