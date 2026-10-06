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


if __name__ == '__main__':
    model = LinearRegression()
    model.weights = np.array([1.0])
    model.bias = 0.0
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([2.0, 3.0, 5.0])
    print(model.compute_gradients(X, y))
