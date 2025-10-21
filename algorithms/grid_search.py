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
        # If v is a tuple/list of two ints, treat as range
        if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, int) for x in v):
            low, high = v
            if n_per_cont > 1:
                vals = np.linspace(low, high, n_per_cont)
                vals = [int(round(x)) for x in vals]
                # Ensure endpoints are included and values are unique
                vals = sorted(set([low] + vals + [high]))
                return vals
            else:
                return [low, high]
        # If v is a list of values, use as is
        return list(v)
    # continuous
    if isinstance(v, (list, tuple)) and len(v) == 2:
        low, high = v
        if log:
            return list(np.exp(np.linspace(math.log(low), math.log(high), n_per_cont)))
        return list(np.linspace(low, high, n_per_cont))
    return [v]

class GridSearchOptimizer(BaseOptimizer):
    def __init__(self, search_space: SearchSpace, n_per_cont: int = 10):
        self.space = search_space
        self.n_per_cont = n_per_cont
        self._grid = None
        self._idx = 0
        self._build_grid()

    def _build_grid(self):
        params = self.space.parameters
        keys = list(params.keys())
        lists = []
        for k in keys:
            expanded = _expand_values(k, params[k], self.n_per_cont)
            print(f"[GridSearch] Expanded values for {k}: {expanded}")
            lists.append(expanded)
        combos = []
        for tup in itertools.product(*lists):
            combos.append({k: tup[i] for i, k in enumerate(keys)})
        import random
        random.shuffle(combos)
        self._grid = combos

    def suggest(self, n: int = 1) -> List[Candidate]:
        """
        Suggest n candidates from the grid. If n is None, default to 1.
        Print parameter sets for debugging.
        """
        if n is None:
            n = 1
        out = []
        for _ in range(n):
            if self._idx >= len(self._grid):
                break
            params = self._grid[self._idx]
            print(f"[GridSearch] Iter {self._idx+1}: {params}")
            out.append(Candidate(params))
            self._idx += 1
        return out

    def update(self, results):
        # grid is deterministic; can store results externally for ranking
        pass
