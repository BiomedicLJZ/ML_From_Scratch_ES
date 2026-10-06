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

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def compute_cost(self, X, y):
        eps = 1e-15
        p = np.clip(self.predict_proba(X), eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


if __name__ == '__main__':
    model = LogisticRegression()
    model.weights = np.array([10.0])
    model.bias = 0.0
    X = np.array([[10.0], [-10.0]])
    y_correct = np.array([1.0, 0.0])
    y_wrong = np.array([0.0, 1.0])
    print(model.compute_cost(X, y_correct))  # Deberia acercarse a 0
    print(model.compute_cost(X, y_wrong))  # Deberia ser grande (~34.5)
