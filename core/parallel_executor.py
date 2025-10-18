# core/parallel_executor.py
from multiprocessing import Pool
from typing import List
from core.base_optimizer import Candidate
import os

def _eval_candidate_worker(args):
    candidate, wrapper, X, y = args
    # wrapper.train_and_score returns cost (lower is better)
    score = wrapper.train_and_score(candidate.params, X, y)
    candidate.score = score
    return candidate

class ParallelExecutor:
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or max(1, os.cpu_count() - 1)

    def evaluate(self, candidates: List[Candidate], wrapper, X, y):
        args = [(cand, wrapper, X, y) for cand in candidates]
        with Pool(self.num_workers) as p:
            results = p.map(_eval_candidate_worker, args)
        return results
