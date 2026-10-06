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


if __name__ == '__main__':
    X = np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [0.2, 0.2], [10.0, 10.0]])
    model = DBSCAN(eps=0.5, min_samples=3)
    labels = np.full(5, -1, dtype=int)
    neighbors = model._region_query(X, 0)
    model._expand_cluster(X, labels, 0, neighbors, 0)
    print(labels)  # Deberia imprimir [ 0  0  0  0 -1]
