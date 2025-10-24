import random
import math
from typing import Dict, Any, Iterable, List, Optional
import numpy as np

class SearchSpace:
    def __init__(self, parameters: Optional[Dict[str, Dict[str, Any]]] = None):
        self.parameters: Dict[str, Dict[str, Any]] = parameters or {}

    def add(self, name: str, param_type: str, values, log: bool = False):
        assert param_type in ("continuous", "discrete", "categorical")
        self.parameters[name] = {"type": param_type, "values": values, "log": log}

    def sample(self) -> Dict[str, Any]:
        out = {}
        for name, info in self.parameters.items():
            t, v, log = info["type"], info["values"], info["log"]

            if t == "continuous":
                low, high = v
                if log:
                    low = max(low, 1e-12)
                    out[name] = float(math.exp(random.uniform(math.log(low), math.log(high))))
                else:
                    out[name] = float(random.uniform(low, high))

            elif t == "discrete":
                # accept either range tuple (low, high) or explicit list/tuple of choices
                if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                    low, high = int(v[0]), int(v[1])
                    out[name] = int(random.randint(low, high))
                else:
                    out[name] = random.choice(v)

            else:  # categorical
                out[name] = random.choice(v)

        return out

    def grid_sample(self, n_per_cont: int = 5, max_configs: Optional[int] = 10000) -> List[Dict[str, Any]]:
        """
        Hybrid grid expansion:
         - categorical: full expansion
         - discrete: if given as (low,high) expand all integers in range; else expand listed values
         - continuous: sample n_per_cont points on linear or log grid
        If total configs > max_configs and max_configs is not None, return a random subset of size max_configs.
        """
        grids = [{}]
        for name, info in self.parameters.items():
            t, v, log = info["type"], info["values"], info["log"]
            new_grids = []

            if t == "categorical":
                choices = list(v)
                for g in grids:
                    for val in choices:
                        ng = g.copy()
                        ng[name] = val
                        new_grids.append(ng)

            elif t == "discrete":
                if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                    low, high = int(v[0]), int(v[1])
                    vals = list(range(low, high + 1))
                else:
                    vals = list(v)
                for g in grids:
                    for val in vals:
                        ng = g.copy()
                        ng[name] = val
                        new_grids.append(ng)

            else:  # continuous
                low, high = v
                if log:
                    low = max(low, 1e-12)
                    vals = list(np.exp(np.linspace(math.log(low), math.log(high), n_per_cont)))
                else:
                    vals = list(np.linspace(low, high, n_per_cont))
                for g in grids:
                    for val in vals:
                        ng = g.copy()
                        ng[name] = float(val)
                        new_grids.append(ng)

            grids = new_grids

            # quick protection: if growth explodes stop early and sample subset
            if len(grids) > max_configs and max_configs is not None:
                # sample subset without replacement
                return random.sample(grids, max_configs)

        # final safety: if still too big, randomly sample
        if max_configs is not None and len(grids) > max_configs:
            return random.sample(grids, max_configs)

        return grids

