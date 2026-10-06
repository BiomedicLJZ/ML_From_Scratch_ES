import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def predict(self, X):
        return X.dot(self.weights) + self.bias

    def compute_cost(self, X, y):
        predictions = self.predict(X)
        errors = predictions - y
        return np.mean(errors ** 2)

    def compute_gradients(self, X, y):
        n_samples = X.shape[0]
        predictions = self.predict(X)
        errors = predictions - y
        dw = (2 / n_samples) * X.T.dot(errors)
        db = (2 / n_samples) * np.sum(errors)
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
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    print(model.weights, model.bias)  # Deberia acercarse a [2.0] 1.0
