from sklearn.neighbors import KNeighborsClassifier
from optiflow.core.search_space import SearchSpace
from optiflow.core.model_wrapper import ModelWrapper

class KNNConfig:
    name = "knn"

    @staticmethod
    def build_search_space():
        s = SearchSpace()
        s.add("n_neighbors", "discrete", list(range(1, 31)))
        s.add("weights", "categorical", ["uniform", "distance"])
        s.add("algorithm", "categorical", ["auto", "ball_tree", "kd_tree", "brute"])
        s.add("leaf_size", "discrete", [5, 150])
        s.add("p", "discrete", [1, 5])
        s.add("metric", "categorical", ["minkowski", "manhattan", "euclidean", "chebyshev"])
        return s

    @staticmethod
    def get_wrapper():
        return ModelWrapper(KNeighborsClassifier)
