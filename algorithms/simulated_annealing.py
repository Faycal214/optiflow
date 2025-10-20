import random
import math
from core.base_optimizer import BaseOptimizer, Candidate
from core.search_space import SearchSpace

class SimulatedAnnealingOptimizer(BaseOptimizer):
    def __init__(self, space: SearchSpace, population_size=10, initial_temp=1.0, cooling_rate=0.9, mutation_rate=0.3):
        super().__init__()
        self.space = space
        self.population = [Candidate(self.space.sample()) for _ in range(population_size)]
        self.scores = {}
        self.temperature = initial_temp
        self.cooling_rate = cooling_rate
        self.mutation_rate = mutation_rate  # <-- add this

    def suggest(self, n=None):
        unevaluated = [c for c in self.population if id(c) not in self.scores]
        if not unevaluated:
            return random.sample(self.population, k=min(len(self.population), n or len(self.population)))
        return unevaluated

    def update(self, results):
        for c in results:
            self.scores[id(c)] = c.score

        best = min(results, key=lambda c: c.score)
        new_population = []
        for c in self.population:
            new_params = self._perturb(c.params)
            new_candidate = Candidate(new_params)
            new_population.append(new_candidate)

        # Acceptance based on simulated annealing
        for new_c, old_c in zip(new_population, self.population):
            old_score = self.scores.get(id(old_c), float("inf"))
            new_score = self.scores.get(id(new_c), old_score)
            delta = new_score - old_score
            if delta < 0 or math.exp(-delta / self.temperature) > random.random():
                old_c.params = new_c.params
                old_c.score = new_score

        self.temperature *= self.cooling_rate

    def _perturb(self, params):
        new_params = params.copy()
        for name, info in self.space.parameters.items():
            if random.random() < self.mutation_rate:
                t = info["type"]
                if t == "categorical":
                    new_params[name] = random.choice(info["values"])
                elif t == "discrete":
                    vals = info["values"]
                    if isinstance(vals, (list, tuple)):
                        new_params[name] = random.choice(vals)
                    else:
                        low, high = vals
                        new_params[name] = random.randint(low, high)
                elif t == "continuous":
                    low, high = info["values"]
                    log = info.get("log", False)
                    if log and (low > 0 and high > 0):
                        log_low, log_high = math.log(low), math.log(high)
                        new_val = math.exp(random.uniform(log_low, log_high))
                    else:
                        new_val = random.uniform(low, high)
                    new_params[name] = new_val
        return new_params
