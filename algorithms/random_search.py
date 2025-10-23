# algorithms/random_search.py
import random
from typing import List

class RandomSearchOptimizer:
    class Candidate:
        def __init__(self, params):
            self.params = params
            self.score = None
            self.model = None

    def __init__(self, search_space, metric, model_class, X, y, n_samples=5000, seed=None):
        self.search_space = search_space
        self.metric = metric
        self.model_class = model_class
        self.X = X
        self.y = y
        self.n_samples = n_samples
        self.rng = random.Random(seed)
        self.best_candidate = None
        self.iteration = 0

    def initialize_population(self):
        self.iteration = 0
        self.best_candidate = None

    def generate_candidates(self):
        return [self.Candidate(self.search_space.sample()) for _ in range(self.n_samples)]

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
        print(f"[Engine] Optimization finished in {time.time()-start_time:.2f}s")
        if self.best_candidate is not None:
            return self.best_candidate.params, self.best_candidate.score
        return None, None
