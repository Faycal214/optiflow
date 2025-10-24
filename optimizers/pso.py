import random
import numpy as np

class PSOOptimizer:
    def __init__(self, search_space, metric, model_class, X, y, n_particles=20, w=0.7, c1=1.4, c2=1.4, velocity_threshold=1e-3, stagnation_limit=10):
        self.search_space = search_space
        self.metric = metric
        self.model_class = model_class
        self.X = X
        self.y = y
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.velocity_threshold = velocity_threshold
        self.stagnation_limit = stagnation_limit
        self.param_info = self.search_space.parameters
        self.particles = []
        self.global_best_position = None
        self.global_best_score = float('-inf')
        self._no_improve_count = 0
        self.iteration = 0
        self.best_params = None

    class Particle:
        def __init__(self, position, velocity):
            self.position = position
            self.velocity = velocity
            self.best_position = position.copy()
            self.best_score = float('-inf')

    def initialize_population(self):
        self.particles = []
        for _ in range(self.n_particles):
            p = self.search_space.sample()
            position = self._encode_position(p)
            velocity = np.zeros_like(position)
            self.particles.append(self.Particle(position, velocity))
        self.global_best_position = None
        self.global_best_score = float('-inf')
        self._no_improve_count = 0
        self.iteration = 0
        self.best_params = None

    def _encode_position(self, param_dict):
        vec = []
        for name, info in self.param_info.items():
            if info["type"] == "categorical":
                vec.append(float(info["values"].index(param_dict[name])))
            else:
                vec.append(float(param_dict[name]))
        return np.array(vec)

    def _decode_position(self, vec):
        params = {}
        for i, (name, info) in enumerate(self.param_info.items()):
            val = vec[i]
            if info["type"] == "categorical":
                idx = int(np.clip(round(val), 0, len(info["values"]) - 1))
                params[name] = info["values"][idx]
            elif info["type"] == "discrete":
                low, high = info["values"]
                params[name] = int(np.clip(round(val), low, high))
            else:
                low, high = info["values"]

        params[name] = float(np.clip(val, low, high))
        return params

    def evaluate_particles(self):
        for particle in self.particles:
            params = self._decode_position(particle.position)
            try:
                model = self.model_class(**params)
                model.fit(self.X, self.y)
                preds = model.predict(self.X)
                if callable(self.metric):
                    score = self.metric(self.y, preds)
                else:
                    from sklearn.metrics import get_scorer
                    score = get_scorer(self.metric)(model, self.X, self.y)
            except Exception:
                score = float('-inf')
            if score > particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = particle.position.copy()
                self.best_params = params

    def update_particles(self):
        for particle in self.particles:
            r1 = np.random.rand(len(particle.position))
            r2 = np.random.rand(len(particle.position))
            cognitive = self.c1 * r1 * (particle.best_position - particle.position)
            social = self.c2 * r2 * (self.global_best_position - particle.position)
            particle.velocity = self.w * particle.velocity + cognitive + social
            particle.position += particle.velocity


        # Find best candidate by evaluating all particles
        best_params = None
        best_score = float('-inf')
        for particle in self.particles:
            params = self._decode_position(particle.position)
            try:
                model = self.model_class(**params)
                model.fit(self.X, self.y)
                preds = model.predict(self.X)
                if callable(self.metric):
                    score = self.metric(self.y, preds)
                else:
                    from sklearn.metrics import get_scorer
                    score = get_scorer(self.metric)(model, self.X, self.y)
            except Exception:
                score = float('-inf')
            if score > best_score:
                best_score = score
                best_params = params
        if best_params is not None:
            print(f"Result for pso: score={best_score:.4f}, params={best_params}")
            return best_params, best_score
        else:
            print("Result for pso: score=-inf, params={}")
            return {}, float('-inf')

    def _update_state(self, candidates, scores=None):
        # Update personal and global bests
        for p, cand in zip(self.particles, candidates):
            score = cand.score
            if score > p.best_score:
                p.best_score = score
                p.best_position = p.position.copy()
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = p.position.copy()

        # Update velocities and positions
        for p in self.particles:
            r1, r2 = np.random.rand(len(p.position)), np.random.rand(len(p.position))
            cognitive = self.c1 * r1 * (p.best_position - p.position)
            social = self.c2 * r2 * (self.global_best_position - p.position)
            p.velocity = self.w * p.velocity + cognitive + social
            p.position += p.velocity

        # Stagnation tracking
        best_score = self.global_best_score
        if self.global_best_score > (getattr(self, '_prev_best_score', float('-inf'))):
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1
        self._prev_best_score = self.global_best_score

    def _check_stop_condition(self, max_iter=None):
        # Stop if velocity below threshold or stagnation or max_iter
        if max_iter is not None and self.iteration >= max_iter:
            return True
        velocities = [np.linalg.norm(p.velocity) for p in self.particles]
        if all(v < self.velocity_threshold for v in velocities):
            return True
        if self._no_improve_count >= self.stagnation_limit:
            return True
        return False

    def _log_progress(self, i, scores, time_elapsed):
        best_score = max(scores) if scores else None
        print(f"[Engine] Iter {i+1} | Best={best_score:.4f} | Time={time_elapsed:.2f}s")




    def update(self, results):
        # Update personal and global bests
        for p, cand in zip(self.particles, results):
            score = cand.score
            if p.best_position is None:
                p.best_position = p.position.copy()
                p.best_score = score
            if score > p.best_score:
                p.best_score = score
                p.best_position = p.position.copy()
            if self.global_best_position is None or score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = p.position.copy()

        # Only update if global_best_position is set
        for p in self.particles:
            if self.global_best_position is None or p.best_position is None:
                continue
            r1, r2 = np.random.rand(len(p.position)), np.random.rand(len(p.position))
            cognitive = self.c1 * r1 * (p.best_position - p.position)
            social = self.c2 * r2 * (self.global_best_position - p.position)
            p.velocity = self.w * p.velocity + cognitive + social
            p.position += p.velocity

    def get_best_params(self):
        return self._decode_position(self.global_best_position)
    
    def run(self, max_iters=10):
        import time
        self.initialize_population()
        start_time = time.time()
        self.best_candidate = None
        for i in range(max_iters):
            self.evaluate_particles()
            # Find best candidate in current population
            best_score = float('-inf')
            best_params = None
            for particle in self.particles:
                params = self._decode_position(particle.position)
                try:
                    model = self.model_class(**params)
                    model.fit(self.X, self.y)
                    preds = model.predict(self.X)
                    if callable(self.metric):
                        score = self.metric(self.y, preds)
                    else:
                        from sklearn.metrics import get_scorer
                        score = get_scorer(self.metric)(model, self.X, self.y)
                except Exception:
                    score = float('-inf')
                if score > best_score:
                    best_score = score
                    best_params = params
            # Update best_candidate
            if self.best_candidate is None or best_score > self.best_candidate.score:
                class Candidate:
                    def __init__(self, params, score):
                        self.params = params
                        self.score = score
                self.best_candidate = Candidate(best_params, best_score)
            self.update_particles()
            print(f"[Engine] Iter {i+1}/{max_iters} | Best={self.best_candidate.score:.4f} | Time={time.time()-start_time:.2f}s\n")
            if self._no_improve_count >= self.stagnation_limit:
                print("[Engine] Stopping early due to stagnation.")
                break
        print(f"[Engine] Optimization finished in {time.time()-start_time:.2f}s")
        if self.best_candidate is not None:
            return self.best_candidate.params, self.best_candidate.score
        else:
            return {}, float('-inf')