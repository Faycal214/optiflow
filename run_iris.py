# examples/run_iris.py
from sklearn.datasets import load_iris
from core.optimization_engine import OptimizationEngine

iris = load_iris()
X = iris.data
y = iris.target

engine = OptimizationEngine(model_key="svc", dataset=(X,y))
model, params, score = engine.run(max_iters=6)
print("best params:", params)
print("best score (negative cv accuracy):", score)
