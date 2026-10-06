import numpy as np


class KMeans:
    def __init__(self, n_clusters=2, n_iterations=100):
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.centroids_ = None
        self.labels_ = None

    def _assign_clusters(self, X, centroids):
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels, n_clusters):
        return np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])


if __name__ == '__main__':
    X = np.array([[0.0, 0.0], [0.2, 0.0], [10.0, 10.0], [10.2, 10.0]])
    labels = np.array([0, 0, 1, 1])
    model = KMeans()
    print(model._update_centroids(X, labels, 2))
