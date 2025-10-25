# optimizers/random_search.py
import random
from typing import List


class RandomSearchOptimizer:
    """Random Search optimization algorithm for hyperparameter tuning.

    Randomly samples configurations from a defined search space, evaluates each
    by fitting a model, and retains the best-performing configuration according
    to a specified metric.

    Attributes:
        search_space: An object defining the parameter distributions to sample from.
        metric: A callable or string name of a sklearn scoring metric.
        model_class: The model constructor (e.g., sklearn estimator class).
        X: Training features.
        y: Training targets.
        n_samples: Number of parameter configurations to sample per iteration.
        rng: Random number generator for reproducibility.
        best_candidate: The best performing candidate found so far.
        iteration: Current optimization iteration count.
        stagnation_limit: Number of iterations without improvement before early stopping.
        _no_improve_count: Counter tracking consecutive non-improving iterations.
    """

    class Candidate:
        """Encapsulates a single parameter configuration and its evaluation result."""

        def __init__(self, params):
            """Initialize a candidate.

            Args:
                params (dict): Parameter configuration for the model.
            """
            self.params = params
            self.score = None
            self.model = None

    def __init__(self, search_space, metric, model_class, X, y, n_samples=10000, seed=None, stagnation_limit=10):
        """Initialize the Random Search optimizer.

        Args:
            search_space: Object defining the parameter search space.
            metric: Callable or sklearn metric name used for evaluation.
            model_class: Model class to be instantiated (e.g., sklearn estimator).
            X: Training features.
            y: Training targets.
            n_samples (int, optional): Number of configurations per iteration. Defaults to 10000.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            stagnation_limit (int, optional): Early stopping limit for no improvement. Defaults to 10.
        """
        self.search_space = search_space
        self.metric = metric
        self.model_class = model_class
        self.X = X
        self.y = y
        self.n_samples = n_samples
        self.rng = random.Random(seed)
        self.best_candidate = None
        self.iteration = 0
        self.stagnation_limit = stagnation_limit
        self._no_improve_count = 0

    def initialize_population(self):
        """Reset optimizer state before running the search."""
        self.iteration = 0
        self.best_candidate = None
        self._no_improve_count = 0

    def generate_candidates(self):
        """Generate a list of randomly sampled candidate parameter sets.

        Returns:
            List[RandomSearchOptimizer.Candidate]: List of new candidates to evaluate.
        """
        return [self.Candidate(self.search_space.sample()) for _ in range(self.n_samples)]

    def evaluate_candidates(self, candidates):
        """Evaluate all candidates by training and scoring models.

        Args:
            candidates (List[RandomSearchOptimizer.Candidate]): Candidates to evaluate.
        """
        for cand in candidates:
            try:
                model = self.model_class(**cand.params)
                model.fit(self.X, self.y)
                preds = model.predict(self.X)
                if callable(self.metric):
                    score = self.metric(self.y, preds)
                else:
                    from sklearn.metrics import get_scorer
                    score = get_scorer(self.metric)(model, self.X, self.y)
                cand.score = score
                cand.model = model
            except Exception:
                cand.score = float('-inf')
                cand.model = None

    def update_state(self, candidates):
        """Update optimizer state with best candidate from the evaluated batch.

        Args:
            candidates (List[RandomSearchOptimizer.Candidate]): Evaluated candidates.
        """
        best = max(candidates, key=lambda c: c.score if c.score is not None else float('-inf'))
        if self.best_candidate is None or best.score > self.best_candidate.score:
            self.best_candidate = best
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

    def run(self, max_iters=10):
        """Run the Random Search optimization loop.

        Args:
            max_iters (int, optional): Maximum number of iterations. Defaults to 10.

        Returns:
            tuple: (best_params, best_score)
                best_params (dict): Parameters of the best model found.
                best_score (float): Corresponding metric score.
        """
        import time
        self.initialize_population()
        start_time = time.time()
        for i in range(max_iters):
            candidates = self.generate_candidates()
            self.evaluate_candidates(candidates)
            self.update_state(candidates)
            print(f"[Engine] Iter {i+1}/{max_iters} | Best={self.best_candidate.score:.4f} | Time={time.time()-start_time:.2f}s")
            self.iteration += 1
            if self._no_improve_count >= self.stagnation_limit:
                print("[Engine] Stopping early due to stagnation.")
                break
        print(f"[Engine] Optimization finished in {time.time()-start_time:.2f}s")
        if self.best_candidate is not None:
            return self.best_candidate.params, self.best_candidate.score
        return None, None
