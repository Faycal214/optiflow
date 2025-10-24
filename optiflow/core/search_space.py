import random, math
import numpy as np

class SearchSpace:
    def __init__(self, parameters=None):
        self.parameters = parameters or {}

    def add(self, name, param_type, values, log=False):
        assert param_type in ("continuous", "discrete", "categorical")
        self.parameters[name] = {"type": param_type, "values": values, "log": log}

    def sample(self):
        out = {}
        for name, info in self.parameters.items():
            t, v, log = info["type"], info["values"], info["log"]

            if t == "continuous":
                low, high = v
                if log:
                    low = max(low, 1e-6)
                    out[name] = math.exp(random.uniform(math.log(low), math.log(high)))
                else:
                    out[name] = random.uniform(low, high)

            elif t == "discrete":
                # Handle both (low, high) or list of possible values
                if (
                    isinstance(v, (list, tuple))
                    and len(v) == 2
                    and all(isinstance(x, (int, float)) for x in v)
                ):
                    low, high = v
                    out[name] = random.randint(low, high)
                else:
                    out[name] = random.choice(v)

            else:  # categorical
                out[name] = random.choice(v)

        return out

    def grid_sample(self, n_per_cont=5):
        """Hybrid: expand categorical/discrete fully and sample continuous on grid."""
        grids = [{}]
        for name, info in self.parameters.items():
            t, v, log = info["type"], info["values"], info["log"]
            new_grids = []

            if t == "categorical":
                for g in grids:
                    for val in v:
                        ng = g.copy()
                        ng[name] = val
                        new_grids.append(ng)

            elif t == "discrete":
                if (
                    isinstance(v, (list, tuple))
                    and len(v) == 2
                    and all(isinstance(x, (int, float)) for x in v)
                ):
                    low, high = v
                    vals = range(low, high + 1)
                else:
                    vals = v
                for g in grids:
                    for val in vals:
                        ng = g.copy()
                        ng[name] = val
                        new_grids.append(ng)

            else:  # continuous
                low, high = v
                if log:
                    vals = np.exp(np.linspace(math.log(low), math.log(high), n_per_cont))
                else:
                    vals = np.linspace(low, high, n_per_cont)
                for g in grids:
                    for val in vals:
                        ng = g.copy()
                        ng[name] = float(val)
                        new_grids.append(ng)

            grids = new_grids
        return grids
