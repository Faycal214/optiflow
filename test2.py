from sklearn.datasets import load_iris, make_classification
from optiflow.optimizers.genetic import GeneticOptimizer
from optiflow.optimizers.pso import PSOOptimizer
from optiflow.optimizers.bayesian import BayesianOptimizer
from optiflow.optimizers.tpe import TPEOptimizer
from optiflow.optimizers.random_search import RandomSearchOptimizer
from optiflow.optimizers.simulated_annealing import SimulatedAnnealingOptimizer
from optiflow.models.configs.random_forest_config import RandomForestConfig

# Custom metric example (macro F1)
def custom_macro_f1(y_true, y_pred):
    
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average="macro")

X, y = make_classification(n_samples = 10000,n_features=10)
search_space = RandomForestConfig.build_search_space()
model_class = RandomForestConfig.get_wrapper().model_class

optimizers = [
    # ("genetic", GeneticOptimizer, {"population": 20, "mutation_prob": 0.3}),
    # ("pso", PSOOptimizer, {"n_particles": 20, "w": 0.7, "c1": 1.4, "c2": 1.4}),
    # ("bayesian", BayesianOptimizer, {"n_initial_points": 5}),
    # ("tpe", TPEOptimizer, {"population_size": 10}),
    # ("random_search", RandomSearchOptimizer, {"n_samples": 20}),
    ("simulated_annealing", SimulatedAnnealingOptimizer, {"population_size": 10, "initial_temp": 1.0, "cooling_rate": 0.9, "mutation_rate": 0.3}),
]

for opt_name, opt_class, opt_params in optimizers:
    print(f"\n{'='*30}\nTesting optimizer: {opt_name}\n{'='*30}")
    optimizer = opt_class(search_space, custom_macro_f1, model_class, X, y, **opt_params)
    best_params, best_score = optimizer.run(max_iters=5)
    print(f"\nResult for {opt_name}: score={best_score:.4f}, params={best_params}")