from multiprocessing import Pool
from typing import List
from optiflow.core.base_optimizer import Candidate
import os

def _eval_candidate_worker(args):
    candidate, wrapper, X, y, scoring = args
    try:
        score, model = wrapper.train_and_score(candidate.params, X, y, scoring=scoring)
        candidate.model = model
        candidate.score = score
    except Exception as e:
        candidate.score = float("-inf")
        candidate.model = None
        print(f"[WARN] Evaluation failed: {e}")
    return candidate

class ParallelExecutor:
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or max(1, os.cpu_count() - 1)

    def evaluate(self, candidates: List[Candidate], wrapper, X, y, scoring="accuracy"):
        args = [(cand, wrapper, X, y, scoring) for cand in candidates]
        with Pool(self.num_workers) as p:
            results = p.map(_eval_candidate_worker, args)
        return results
