# algorithms/grid_search.py
import itertools
from typing import List
from core.base_optimizer import BaseOptimizer, Candidate
from core.search_space import SearchSpace
import math
import numpy as np

def _expand_values(name, info, n_per_cont=5):
    t, v, log = info["type"], info["values"], info.get("log", False)
    if t == "categorical":
        return list(v)
    if t == "discrete":
        if isinstance(v, (list, tuple)):
            # explicit list
            return list(v)
        low, high = v
        # limit grid size
        high = min(high, low + 200)
        return list(range(low, high + 1))
    # continuous
    low, high = v
    if log:
        return list(np.exp(np.linspace(math.log(low), math.log(high), n_per_cont)))
    return list(np.linspace(low, high, n_per_cont))

class GridSearchOptimizer(BaseOptimizer):
    def __init__(self, search_space: SearchSpace, n_per_cont: int = 5):
        self.space = search_space
        self.n_per_cont = n_per_cont
        self._grid = None
        self._idx = 0
        self._build_grid()

    def _build_grid(self):
        params = self.space.parameters
        keys = list(params.keys())
        lists = [_expand_values(k, params[k], self.n_per_cont) for k in keys]
        combos = []
        for tup in itertools.product(*lists):
            combos.append({k: tup[i] for i, k in enumerate(keys)})
        self._grid = combos

    def suggest(self, n: int = 1) -> List[Candidate]:
        out = []
        for _ in range(n):
            if self._idx >= len(self._grid):
                break
            out.append(Candidate(self._grid[self._idx]))
            self._idx += 1
        return out

    def update(self, results):
        # grid is deterministic; can store results externally for ranking
        pass
