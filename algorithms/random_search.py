# algorithms/random_search.py
import random
from typing import List
from core.base_optimizer import BaseOptimizer, Candidate
from core.search_space import SearchSpace

class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, search_space: SearchSpace, n_samples: int = 50, seed: int = None):
        self.space = search_space
        self.n_samples = n_samples
        self.rng = random.Random(seed)

    def suggest(self, n: int = None) -> List[Candidate]:
        n = n or self.n_samples
        return [Candidate(self.space.sample()) for _ in range(n)]

    def update(self, results):
        # Random search is stateless. Keep results if you want logging externally.
        pass
