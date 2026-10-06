import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def predict(self, X):
        return X.dot(self.weights) + self.bias


if __name__ == '__main__':
    model = LinearRegression()
    model.weights = np.array([2.0, -1.0])
    model.bias = 0.5
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    print(model.predict(X))  # Deberia imprimir [0.5 2.5]
