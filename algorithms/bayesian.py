# algorithms/bayesian.py
import random
from typing import List

from skopt import Optimizer as SkOptimizer
from skopt.space import Real, Integer, Categorical

class BayesianOptimizer:
    class Candidate:
        def __init__(self, params):
            self.params = params
            self.score = None
            self.model = None

    def __init__(self, search_space, metric, model_class, X, y, n_initial_points=5, random_state=None, stagnation_limit=10):
        self.search_space = search_space
        self.metric = metric
        self.model_class = model_class
        self.X = X
        self.y = y
        self.n_initial_points = n_initial_points
        self.random_state = random_state or 42
        self.sk_space = self._to_skopt_space()
        self.optimizer = SkOptimizer(
            dimensions=self.sk_space,
            random_state=self.random_state,
            n_initial_points=self.n_initial_points
        )
        self.trials = []
        self.results = []
        self.iteration = 0
        self.best_candidate = None
        self.stagnation_limit = stagnation_limit
        self._no_improve_count = 0

    def initialize_population(self):
        self.trials = []
        self.results = []
        self.iteration = 0
        self.best_candidate = None
        self._no_improve_count = 0

    def generate_candidates(self):
        if len(self.trials) == 0:
            samples = [self.search_space.sample() for _ in range(self.n_initial_points)]
            return [self.Candidate(s) for s in samples]
        suggestions = self.optimizer.ask(n_points=1)
        param_names = list(self.search_space.parameters.keys())
        candidates = []
        for s in suggestions:
            cand_params = dict(zip(param_names, s))
            candidates.append(self.Candidate(cand_params))
        return candidates

    def evaluate_candidates(self, candidates):
        for cand in candidates:
            try:
                model = self.model_class(**cand.params)
                model.fit(self.X, self.y)
                preds = model.predict(self.X)
                if callable(self.metric):
                    score = self.metric(self.y, preds)
                else:
                    from sklearn.metrics import get_scorer
                    score = get_scorer(self.metric)(model, self.X, self.y)
                cand.score = score
                cand.model = model
            except Exception:
                cand.score = float('-inf')
                cand.model = None

    def update_state(self, candidates):
        for cand in candidates:
            x = [cand.params[k] for k in self.search_space.parameters.keys()]
            y = -cand.score
            x_clean = []
            for xi, (name, info) in zip(x, self.search_space.parameters.items()):
                t = info["type"]
                v = info["values"]
                if t in ["continuous", "int"]:
                    low, high = v
                    xi = max(min(xi, high), low)
                elif t in ["categorical", "discrete"]:
                    pass
            # Stagnation logic
            if self.best_candidate is None or cand.score > self.best_candidate.score:
                self.best_candidate = cand
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
            # Properly encode x_clean for skopt
            x_clean = []
            for name, info in self.search_space.parameters.items():
                val = cand.params[name]
                t = info["type"]
                v = info["values"]
                # If categorical/discrete and val is None, replace with 'none' or first valid value
                if (t == "categorical" or t == "discrete"):
                    if val is None:
                        val = 'none' if 'none' in v else v[0]
                    # If value not in allowed values, use first valid value
                    if val not in v:
                        val = v[0]
                x_clean.append(val)
            self.trials.append((x_clean, y))
            self.optimizer.tell(x_clean, y)
        self.results.extend(candidates)
        best = max(candidates, key=lambda c: c.score if c.score is not None else float('-inf'))
        if self.best_candidate is None or best.score > self.best_candidate.score:
            self.best_candidate = best

    def run(self, max_iters=10):
        import time
        self.initialize_population()
        start_time = time.time()
        for i in range(max_iters):
            candidates = self.generate_candidates()
            self.evaluate_candidates(candidates)
            self.update_state(candidates)
            scores = [c.score for c in candidates]
            print(f"[Engine] Iter {i+1}/{max_iters} | Best={self.best_candidate.score:.4f} | Time={time.time()-start_time:.2f}s")
            self.iteration += 1
            if self._no_improve_count >= self.stagnation_limit:
                print("[Engine] Stopping early due to stagnation.")
                break
        print(f"[Engine] Optimization finished in {time.time()-start_time:.2f}s")
        if self.best_candidate is not None:
            return self.best_candidate.params, self.best_candidate.score
        return None, None

    def _to_skopt_space(self):
        out = []
        for name, info in self.search_space.parameters.items():
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
