
import optuna

class TPEOptimizer:
    class Candidate:
        def __init__(self, params):
            self.params = params
            self.score = None
            self.model = None
            self.trial = None

    def __init__(self, search_space, metric, model_class, X, y, population_size=10, stagnation_limit=10):
        self.search_space = search_space
        self.metric = metric
        self.model_class = model_class
        self.X = X
        self.y = y
        self.population_size = population_size
        self.study = optuna.create_study(direction="maximize")
        self.trials = []
        self.iteration = 0
        self.best_candidate = None
        self.stagnation_limit = stagnation_limit
        self._no_improve_count = 0

    def initialize_population(self):
        self.trials = []
        self.iteration = 0
        self.best_candidate = None
        self._no_improve_count = 0

    def _define_search_space(self, trial):
        params = {}
        for name, cfg in self.search_space.parameters.items():
            ptype = cfg["type"]
            values = cfg["values"]
            log = cfg.get("log", False)
            if ptype == "int" or ptype == "discrete":
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

    def generate_candidates(self):
        candidates = []
        for _ in range(self.population_size):
            trial = self.study.ask()
            params = self._define_search_space(trial)
            candidate = self.Candidate(params=params)
            candidate.trial = trial
            candidates.append(candidate)
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
        for candidate in candidates:
            score = candidate.score
            trial = getattr(candidate, "trial", None)
            if trial is not None:
                self.study.tell(trial, score)
            # Stagnation logic
            if self.best_candidate is None or score > self.best_candidate.score:
                self.best_candidate = candidate
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
            self.trials.append((candidate.params, score))
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
