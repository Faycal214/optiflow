# core/search_space.py
import random
from typing import Dict, Any

class SearchSpace:
    def __init__(self):
        self.parameters: Dict[str, Dict[str, Any]] = {}

    def add(self, name: str, param_type: str, values):
        assert param_type in ("continuous","discrete","categorical")
        self.parameters[name] = {"type": param_type, "values": values}

    def sample(self):
        out = {}
        for name, info in self.parameters.items():
            if info["type"] == "continuous":
                low, high = info["values"]
                out[name] = random.uniform(low, high)
            elif info["type"] == "discrete":
                low, high = info["values"]
                out[name] = random.randint(low, high)
            else:
                out[name] = random.choice(info["values"])
        return out

    def grid_sample(self):
        # naive: expand only categorical/discrete small spaces (helper)
        raise NotImplementedError("grid_sample not implemented in starter")
