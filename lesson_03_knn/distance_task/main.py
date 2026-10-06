import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))


if __name__ == '__main__':
    model = KNNClassifier()
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    print(model._euclidean_distance(a, b))  # Deberia imprimir 5.0
