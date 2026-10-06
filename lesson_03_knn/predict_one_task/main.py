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

    def _predict_one(self, x):
        distances = np.array([self._euclidean_distance(x, x_train) for x_train in self.X_train])
        nearest_indices = np.argsort(distances)[:self.n_neighbors]
        nearest_labels = self.y_train[nearest_indices]
        return Counter(nearest_labels.tolist()).most_common(1)[0][0]


if __name__ == '__main__':
    model = KNNClassifier(n_neighbors=3)
    X_train = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [10.0, 10.0], [10.1, 10.1], [9.9, 10.0]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    model.fit(X_train, y_train)
    print(model._predict_one(np.array([0.05, 0.05])))  # Deberia imprimir 0
