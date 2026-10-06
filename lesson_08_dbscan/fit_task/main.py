import numpy as np


class DBSCAN:
    def __init__(self, eps=1.0, min_samples=3):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _region_query(self, X, point_idx):
        distances = np.sqrt(((X - X[point_idx]) ** 2).sum(axis=1))
        return np.where(distances <= self.eps)[0]

    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        labels[point_idx] = cluster_id
        queue = list(neighbors)
        visited_in_queue = set(queue)
        i = 0
        while i < len(queue):
            current = queue[i]
            i += 1
            if labels[current] == -1:
                labels[current] = cluster_id
                current_neighbors = self._region_query(X, current)
                if len(current_neighbors) >= self.min_samples:
                    for neighbor in current_neighbors:
                        if neighbor not in visited_in_queue:
                            visited_in_queue.add(neighbor)
                            queue.append(neighbor)

    def fit(self, X):
        n = X.shape[0]
        labels = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0
        for point_idx in range(n):
            if visited[point_idx]:
                continue
            visited[point_idx] = True
            neighbors = self._region_query(X, point_idx)
            if len(neighbors) < self.min_samples:
                continue
            self._expand_cluster(X, labels, point_idx, neighbors, cluster_id)
            cluster_id += 1
            visited[neighbors] = True
        self.labels_ = labels
        return self


if __name__ == '__main__':
    blob_a = np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [0.2, 0.2]])
    blob_b = np.array([[10.0, 10.0], [10.2, 10.0], [10.0, 10.2], [10.2, 10.2]])
    outlier = np.array([[50.0, 50.0]])
    X = np.vstack([blob_a, blob_b, outlier])
    model = DBSCAN(eps=0.5, min_samples=3)
    model.fit(X)
    print(model.labels_)  # Deberia imprimir [ 0  0  0  0  1  1  1  1 -1]
