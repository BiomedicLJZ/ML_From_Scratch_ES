import unittest
import numpy as np
from main import LinearRegression


class TestCase(unittest.TestCase):

    def test_fit_recovers_exact_line(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = 2 * X.flatten() + 1
        model = LinearRegression(learning_rate=0.1, n_iterations=2000)
        model.fit(X, y)
        self.assertAlmostEqual(2.0, model.weights[0], places=2, msg="weights deberia converger cerca de 2.0")
        self.assertAlmostEqual(1.0, model.bias, places=2, msg="bias deberia converger cerca de 1.0")

    def test_fit_reduces_cost(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(-5, 5, size=(50, 2))
        true_w = np.array([3.0, -2.0])
        y = X.dot(true_w) + 5.0
        model = LinearRegression(learning_rate=0.01, n_iterations=500)
        model_before = LinearRegression()
        model_before.weights = np.zeros(2)
        model_before.bias = 0.0
        cost_before = model_before.compute_cost(X, y)
        model.fit(X, y)
        cost_after = model.compute_cost(X, y)
        self.assertLess(cost_after, cost_before, msg="fit deberia reducir el MSE respecto a su valor inicial")

    def test_fit_returns_self(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])
        model = LinearRegression(learning_rate=0.1, n_iterations=10)
        result = model.fit(X, y)
        self.assertIs(model, result, msg="fit deberia devolver self")


if __name__ == '__main__':
    unittest.main()
