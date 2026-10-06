import numpy as np


class DBSCAN:
    def __init__(self, eps=1.0, min_samples=3):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _region_query(self, X, point_idx):
        distances = np.sqrt(((X - X[point_idx]) ** 2).sum(axis=1))
        return np.where(distances <= self.eps)[0]


if __name__ == '__main__':
    X = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 5.0]])
    model = DBSCAN(eps=0.5, min_samples=2)
    print(model._region_query(X, 0))  # Deberia imprimir [0 1]
