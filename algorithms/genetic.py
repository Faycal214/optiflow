# algorithms/genetic.py
import random
import numpy as np
from core.base_optimizer import BaseOptimizer, Candidate

class GeneticOptimizer(BaseOptimizer):
    def __init__(self, search_space, population_size=20, generations=20, mutation_rate=0.2, elite_frac=0.2):
        self.search_space = search_space
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_frac = elite_frac
        self.population = [Candidate(self.search_space.sample()) for _ in range(population_size)]

    def suggest(self, n=None):
        # Return current population for evaluation
        return self.population

    def update(self, results):
        # Sort by fitness (lower score = better)
        sorted_pop = sorted(results, key=lambda c: c.score)
        elite_count = max(1, int(self.elite_frac * len(sorted_pop)))
        elites = sorted_pop[:elite_count]

        # Breed new individuals
        new_population = []
        while len(new_population) < self.pop_size - elite_count:
            p1, p2 = random.sample(elites, 2)
            child_params = self.crossover(p1.params, p2.params)
            child_params = self.mutate(child_params)
            new_population.append(Candidate(child_params))

        self.population = elites + new_population

    def crossover(self, p1, p2):
        child = {}
        for k in p1.keys():
            # 50/50 parameter inheritance
            child[k] = random.choice([p1[k], p2[k]])
        return child

    def mutate(self, params):
        # Randomly perturb selected parameters
        for k, info in self.search_space.parameters.items():
            if random.random() < self.mutation_rate:
                if info["type"] == "continuous":
                    low, high = info["values"]
                    params[k] = random.uniform(low, high)
                elif info["type"] == "discrete":
                    low, high = info["values"]
                    params[k] = random.randint(low, high)
                else:
                    params[k] = random.choice(info["values"])
        return params
