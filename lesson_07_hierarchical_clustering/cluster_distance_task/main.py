import numpy as np


class AgglomerativeClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.labels_ = None

    def _pairwise_distances(self, X):
        return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

    def _cluster_distance(self, cluster_a, cluster_b, distances):
        return distances[np.ix_(cluster_a, cluster_b)].min()


if __name__ == '__main__':
    model = AgglomerativeClustering()
    X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
    distances = model._pairwise_distances(X)
    print(model._cluster_distance([0, 1], [2, 3], distances))
