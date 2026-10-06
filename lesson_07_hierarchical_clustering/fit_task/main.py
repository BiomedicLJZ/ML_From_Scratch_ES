import numpy as np


class AgglomerativeClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.labels_ = None

    def _pairwise_distances(self, X):
        return np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

    def _cluster_distance(self, cluster_a, cluster_b, distances):
        return distances[np.ix_(cluster_a, cluster_b)].min()

    def fit(self, X):
        n = X.shape[0]
        distances = self._pairwise_distances(X)
        clusters = [[i] for i in range(n)]
        while len(clusters) > self.n_clusters:
            best = (None, None, np.inf)
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    d = self._cluster_distance(clusters[a], clusters[b], distances)
                    if d < best[2]:
                        best = (a, b, d)
            a, b, _ = best
            clusters[a] = clusters[a] + clusters[b]
            del clusters[b]
        labels = np.empty(n, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_idx in cluster:
                labels[point_idx] = cluster_id
        self.labels_ = labels
        return self


if __name__ == '__main__':
    X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
    model = AgglomerativeClustering(n_clusters=2)
    model.fit(X)
    print(model.labels_)
