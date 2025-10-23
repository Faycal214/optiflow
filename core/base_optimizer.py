# core/base_optimizer.py
from typing import List, Dict, Any


class Candidate:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.score = None
        self.model = None  # Optionally store trained model
class BaseOptimizer:
    """
    Unified population-based optimizer interface.
    All optimizers should manage a population of Candidate objects and evolve them iteratively.
    """
    def __init__(self, search_space, population_size: int = 10, **kwargs):
        self.search_space = search_space
        self.population_size = population_size
        self.population: List[Candidate] = []
        self.iteration = 0
        self.history: List[List[Candidate]] = []  # Store population per iteration
        self.best_candidate: Candidate = None
        self.params = kwargs  # Store optimizer-specific parameters

    def initialize_population(self):
        """Initialize the population of candidate solutions."""
        self.population = [Candidate(self.search_space.sample()) for _ in range(self.population_size)]
        self.iteration = 0
        self.history = []
        self.best_candidate = None

    def suggest(self, n: int = None) -> List[Candidate]:
        """Return current population or batch of candidates to evaluate."""
        if not self.population:
            self.initialize_population()
        if n is not None:
            return self.population[:n]
        return list(self.population)

    def update(self, results: List[Candidate]):
        """Update population based on evaluated results. To be implemented by subclasses."""
        raise NotImplementedError

    def get_best(self) -> Candidate:
        """Return the best candidate found so far."""
        if not self.population:
            return None
        return max(self.population, key=lambda c: c.score if c.score is not None else float('-inf'))

    def reset(self):
        """Reset optimizer state for a new run."""
        self.initialize_population()

