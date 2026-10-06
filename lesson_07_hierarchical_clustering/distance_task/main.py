import numpy as np


class AgglomerativeClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.labels_ = None

    def _pairwise_distances(self, X):
        return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))


if __name__ == '__main__':
    model = AgglomerativeClustering()
    X = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]])
    print(model._pairwise_distances(X))
