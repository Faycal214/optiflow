import time
import json
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import numpy as np

def optimize_model(model, param_grid, X, y, cv=5, n_jobs=-1, verbose=True):
    results = []
    start_time = time.time()
    total_configs = len(param_grid)
    print(f"Starting optimization on {total_configs} configurations...")

    with tqdm(total=total_configs, desc="Optimization Progress") as pbar:
        for i, params in enumerate(param_grid):
            iter_start = time.time()
            model.set_params(**params)
            
            scores = cross_val_score(model, X, y, cv=cv, n_jobs=n_jobs)
            mean_score = np.mean(scores)
            iter_time = time.time() - iter_start
            
            results.append({
                "iteration": i + 1,
                "params": params,
                "mean_score": mean_score,
                "iteration_time_sec": iter_time
            })

            if verbose:
                print(f"[{i+1}/{total_configs}] Score: {mean_score:.4f} | Time: {iter_time:.2f}s")

            pbar.update(1)

    total_time = time.time() - start_time
    print(f"\nOptimization complete in {total_time:.2f} seconds.")

    best_result = max(results, key=lambda x: x["mean_score"])
    with open("optimization_log.json", "w") as f:
        json.dump({
            "total_time_sec": total_time,
            "results": results,
            "best_result": best_result
        }, f, indent=4)

    print(f"Best score: {best_result['mean_score']:.4f}")
    return best_result, results
