# algorithms/genetic.py
import random
from core.base_optimizer import BaseOptimizer, Candidate

class GeneticOptimizer(BaseOptimizer):
    def __init__(self, search_space, population_size=10, elite_frac=0.2, mutation_prob=0.2):
        self.search_space = search_space
        self.pop_size = population_size
        self.elite_frac = elite_frac
        self.mutation_prob = mutation_prob
        self.population = [Candidate(self.search_space.sample()) for _ in range(self.pop_size)]

    def suggest(self, n=None):
        # return current population to evaluate
        return self.population

    def update(self, results):
        # results: list of Candidate with score filled (lower is better)
        # sort ascending by score
        sorted_pop = sorted(results, key=lambda c: c.score)
        elite_count = max(1, int(self.elite_frac * len(sorted_pop)))
        elites = sorted_pop[:elite_count]

        # produce children
        children = []
        while len(children) < self.pop_size - elite_count:
            a, b = random.sample(elites, 2)
            child_params = self.crossover(a.params, b.params)
            child_params = self.mutate(child_params)
            children.append(Candidate(child_params))

        self.population = elites + children

    def crossover(self, p1, p2):
        child = {}
        for k in p1:
            child[k] = random.choice([p1[k], p2[k]])
        return child

    def mutate(self, params):
        for k in params:
            if random.random() < self.mutation_prob:
                info = self.search_space.parameters[k]
                if info["type"] == "continuous":
                    low, high = info["values"]
                    params[k] = random.uniform(low, high)
                elif info["type"] == "discrete":
                    low, high = info["values"]
                    params[k] = random.randint(low, high)
                else:
                    params[k] = random.choice(info["values"])
        return params
