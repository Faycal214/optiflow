import random
import numpy as np
from core.base_optimizer import BaseOptimizer, Candidate

class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
        self.best_position = position.copy()
        self.best_score = float("inf")

class PSOOptimizer(BaseOptimizer):
    def __init__(self, search_space, n_particles=20, w=0.7, c1=1.4, c2=1.4):
        self.space = search_space
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.particles = []
        self.global_best_position = None
        self.global_best_score = float("inf")
        self.param_info = self.space.parameters
        self.numeric_params = {k: v for k, v in self.param_info.items() if v["type"] in ("continuous", "discrete")}
        self.cat_params = {k: v for k, v in self.param_info.items() if v["type"] == "categorical"}

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

    def _init_particles(self):
        for _ in range(self.n_particles):
            p = self.space.sample()
            position = self._encode_position(p)
            velocity = np.zeros_like(position)
            self.particles.append(Particle(position, velocity))

    def suggest(self, n=None):
        if not self.particles:
            self._init_particles()
        return [Candidate(self._decode_position(p.position)) for p in self.particles]

    def update(self, results):
        for p, cand in zip(self.particles, results):
            score = cand.score
            if score < p.best_score:
                p.best_score = score
                p.best_position = p.position.copy()
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = p.position.copy()

        for p in self.particles:
            r1, r2 = np.random.rand(len(p.position)), np.random.rand(len(p.position))
            cognitive = self.c1 * r1 * (p.best_position - p.position)
            social = self.c2 * r2 * (self.global_best_position - p.position)
            p.velocity = self.w * p.velocity + cognitive + social
            p.position += p.velocity

    def get_best_params(self):
        return self._decode_position(self.global_best_position)
