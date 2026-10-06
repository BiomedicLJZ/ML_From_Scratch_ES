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

    def fit(self, X, initial_centroids):
        centroids = np.array(initial_centroids, dtype=float)
        labels = None
        for _ in range(self.n_iterations):
            labels = self._assign_clusters(X, centroids)
            centroids = self._update_centroids(X, labels, len(centroids))
        self.centroids_ = centroids
        self.labels_ = labels
        return self


if __name__ == '__main__':
    X = np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [10.0, 10.0], [10.2, 10.0], [10.0, 10.2]])
    initial_centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    model = KMeans(n_clusters=2, n_iterations=10)
    model.fit(X, initial_centroids)
    print(model.labels_)
    print(model.centroids_)
