import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        return self._sigmoid(X.dot(self.weights) + self.bias)


if __name__ == '__main__':
    model = LogisticRegression()
    model.weights = np.array([1.0, -1.0])
    model.bias = 0.0
    X = np.array([[2.0, 2.0], [0.0, 5.0]])
    print(model.predict_proba(X))  # Deberia imprimir [0.5 0.00669285]
