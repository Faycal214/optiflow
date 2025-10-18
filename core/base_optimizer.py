# core/base_optimizer.py
from typing import List, Dict, Any

class Candidate:
    def __init__(self, params: Dict[str,Any]):
        self.params = params
        self.score = None

class BaseOptimizer:
    def __init__(self, **kwargs):
        pass

    def suggest(self, n: int):
        """Return list[Candidate] to evaluate."""
        raise NotImplementedError

    def update(self, results):
        """Update internal state from evaluation results."""
        raise NotImplementedError
