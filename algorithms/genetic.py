# algorithms/genetic.py
import random
from typing import List
from core.base_optimizer import BaseOptimizer, Candidate
from core.search_space import SearchSpace
import copy

class GeneticOptimizer(BaseOptimizer):
    def __init__(self, search_space: SearchSpace, population: int = 20, elite_frac: float = 0.2,
                 crossover_prob: float = 0.8, mutation_prob: float = 0.2, seed: int = None):
        self.space = search_space
        self.pop_size = population
        self.elite_frac = elite_frac
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.rng = random.Random(seed)

        # population of Candidate
        self.population: List[Candidate] = [Candidate(self.space.sample()) for _ in range(self.pop_size)]
        # after evaluation each candidate should have .score

    def suggest(self, n: int = None) -> List[Candidate]:
        # if population un-evaluated return initial
        return list(self.population)

    def update(self, results: List[Candidate]):
        # results: list of evaluated Candidate with .score (lower better)
        # sort ascending by cost
        results = sorted(results, key=lambda c: c.score)
        elite_n = max(1, int(self.elite_frac * len(results)))
        elites = results[:elite_n]

        # produce new population
        new_pop = elites.copy()
        while len(new_pop) < self.pop_size:
            parent_a = self._tournament_select(results)
            parent_b = self._tournament_select(results)
            child_params = self._crossover(parent_a.params, parent_b.params)
            child_params = self._mutate(child_params)
            new_pop.append(Candidate(child_params))
        self.population = new_pop

    # ---------------- helpers ----------------
    def _tournament_select(self, population, k=3):
        pick = self.rng.sample(population, min(k, len(population)))
        return min(pick, key=lambda c: c.score)

    def _crossover(self, a: dict, b: dict) -> dict:
        if self.rng.random() > self.crossover_prob:
            return copy.deepcopy(a) if self.rng.random() < 0.5 else copy.deepcopy(b)

        child = {}
        for key in a.keys():
            # uniform crossover
            child[key] = copy.deepcopy(a[key]) if self.rng.random() < 0.5 else copy.deepcopy(b[key])
        return child

    def _mutate(self, params: dict) -> dict:
        new = copy.deepcopy(params)
        for k, info in self.space.parameters.items():
            if self.rng.random() > self.mutation_prob:
                continue
            t = info["type"]
            if t == "categorical":
                new[k] = self.rng.choice(info["values"])
            elif t == "discrete":
                v = info["values"]
                if isinstance(v, (list, tuple)):
                    new[k] = self.rng.choice(v)
                else:
                    low, high = v
                    new[k] = self.rng.randint(low, high)
            else:  # continuous
                low, high = info["values"]
                log = info.get("log", False)
                if log:
                    new[k] = float(self.rng.uniform(math.log(low), math.log(high)))
                    new[k] = math.exp(new[k])
                else:
                    new[k] = float(self.rng.uniform(low, high))
        return new
