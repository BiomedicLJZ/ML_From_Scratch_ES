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

    def compute_gradients(self, X, y):
        n_samples = X.shape[0]
        errors = self.predict_proba(X) - y
        dw = (1 / n_samples) * X.T.dot(errors)
        db = (1 / n_samples) * np.sum(errors)
        return dw, db

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        for _ in range(self.n_iterations):
            dw, db = self.compute_gradients(X, y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self


if __name__ == '__main__':
    X = np.array([[1.0], [2.0], [8.0], [9.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    model = LogisticRegression(learning_rate=0.5, n_iterations=2000)
    model.fit(X, y)
    print(model.predict(X))  # Deberia imprimir [0 0 1 1]
