import random
import math

class SimulatedAnnealingOptimizer:
    class Candidate:
        def __init__(self, params):
            self.params = params
            self.score = None
            self.model = None

    def __init__(self, search_space, metric, model_class, X, y, population_size=10, initial_temp=1.0, cooling_rate=0.9, mutation_rate=0.3, t_min=1e-3):
        self.search_space = search_space
        self.metric = metric
        self.model_class = model_class
        self.X = X
        self.y = y
        self.population_size = population_size
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.mutation_rate = mutation_rate
        self.t_min = t_min
        self.temperature = initial_temp
        self.scores = {}
        self.iteration = 0
        self.population = []

    def initialize_population(self):
        self.population = [self.Candidate(self.search_space.sample()) for _ in range(self.population_size)]
        self.scores = {}
        self.temperature = self.initial_temp
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

    def update_state(self):
        # Store scores
        for c in self.population:
            self.scores[id(c)] = c.score

        # Generate new population by perturbation
        new_population = []
        for c in self.population:
            new_params = self._perturb(c.params)
            new_candidate = self.Candidate(new_params)
            new_population.append(new_candidate)

        # Acceptance based on simulated annealing
        for new_c, old_c in zip(new_population, self.population):
            old_score = self.scores.get(id(old_c), float('inf'))
            new_score = self.scores.get(id(new_c), old_score)
            delta = new_score - old_score
            if delta < 0 or math.exp(-delta / self.temperature) > random.random():
                old_c.params = new_c.params
                old_c.score = new_score

        self.temperature *= self.cooling_rate

    def run(self, max_iters=10):
        import time
        self.initialize_population()
        start_time = time.time()
        best_score = float('-inf')
        self._no_improve_count = 0
        for i in range(max_iters):
            self.evaluate_population()
            self.update_state()
            scores = [c.score for c in self.population]
            current_best = max(scores) if scores else float('-inf')
            print(f"[Engine] Iter {i+1}/{max_iters} | Best={current_best:.4f} | Time={time.time()-start_time:.2f}s")
            self.iteration += 1
            if current_best > best_score:
                best_score = current_best
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
            if self._no_improve_count >= getattr(self, 'stagnation_limit', 10):
                print("[Engine] Stopping early due to stagnation.")
                break
            if self.temperature < self.t_min:
                break
        print(f"[Engine] Optimization finished in {time.time()-start_time:.2f}s")
        best = max(self.population, key=lambda c: c.score if c.score is not None else float('-inf'))
        return best.params, best.score

    def _perturb(self, params):
        new_params = params.copy()
        for name, info in self.search_space.parameters.items():
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
