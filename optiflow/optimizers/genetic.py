# optimizers/genetic.py
import random
import math
from typing import List
from optiflow.core.base_optimizer import BaseOptimizer, Candidate
from optiflow.core.search_space import SearchSpace

# Standalone GeneticOptimizer
import copy

class GeneticOptimizer:
    class Candidate:
        def __init__(self, params):
            self.params = params
            self.score = None
            self.model = None

    def __init__(self, search_space, metric, model_class, X, y, population=20, elite_frac=0.2,
                 crossover_prob=0.8, mutation_prob=0.2, seed=None, stagnation_limit=10):
        self.search_space = search_space
        self.metric = metric
        self.model_class = model_class
        self.X = X
        self.y = y
        self.population_size = population
        self.elite_frac = elite_frac
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.rng = random.Random(seed)
        self.stagnation_limit = stagnation_limit
        self._no_improve_count = 0
        self._best_score = None
        self.population = []
        self.best_candidate = None
        self.iteration = 0

    def initialize_population(self):
        self.population = [self.Candidate(self.search_space.sample()) for _ in range(self.population_size)]
        self._no_improve_count = 0
        self._best_score = None
        self.best_candidate = None
        self.iteration = 0

    def evaluate_population(self):
        for cand in self.population:
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

    def select_elites(self):
        sorted_pop = sorted(self.population, key=lambda c: c.score if c.score is not None else float('-inf'), reverse=True)
        elite_n = max(1, int(self.elite_frac * len(sorted_pop)))
        return sorted_pop[:elite_n]

    def crossover(self, parent1, parent2):
        child_params = {}
        for k in parent1.params:
            child_params[k] = parent1.params[k] if self.rng.random() < 0.5 else parent2.params[k]
        return self.Candidate(child_params)

    def mutate(self, candidate):
        params = candidate.params.copy()
        for k, v in params.items():
            if self.rng.random() < self.mutation_prob:
                params[k] = self.search_space.sample()[k]
        return self.Candidate(params)

    def run(self, max_iters=10):
        import time
        self.initialize_population()
        start_time = time.time()
        for i in range(max_iters):
            self.evaluate_population()
            elites = self.select_elites()
            new_population = elites.copy()
            while len(new_population) < self.population_size:
                if self.rng.random() < self.crossover_prob:
                    p1, p2 = self.rng.sample(elites, 2)
                    child = self.crossover(p1, p2)
                else:
                    child = self.rng.choice(elites)
                child = self.mutate(child)
                new_population.append(child)
            self.population = new_population
            best = max(self.population, key=lambda c: c.score if c.score is not None else float('-inf'))
            if self.best_candidate is None or best.score > self.best_candidate.score:
                self.best_candidate = best
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
            print(f"[Engine] Iter {i+1}/{max_iters} | Best={self.best_candidate.score:.4f} | Time={time.time()-start_time:.2f}s")
            if self._no_improve_count >= self.stagnation_limit:
                print("[Engine] Stopping early due to stagnation.")
                break
        print(f"[Engine] Optimization finished in {time.time()-start_time:.2f}s")
        return self.best_candidate.params, self.best_candidate.score

    def _tournament_select(self, population, k=3):
        pick = self.rng.sample(population, min(k, len(population)))
        return max(pick, key=lambda c: c.score if c.score is not None else float('-inf'))
