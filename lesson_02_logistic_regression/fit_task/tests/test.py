import unittest
import numpy as np
from main import LogisticRegression


class TestCase(unittest.TestCase):

    def test_fit_separates_simple_data(self):
        X = np.array([[1.0], [2.0], [8.0], [9.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        model = LogisticRegression(learning_rate=0.5, n_iterations=2000)
        model.fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y.astype(int), err_msg="Deberia clasificar perfectamente un conjunto de datos claramente separable")

    def test_fit_reduces_cost(self):
        rng = np.random.default_rng(0)
        class_0 = rng.normal(loc=-2.0, scale=0.5, size=(15, 2))
        class_1 = rng.normal(loc=2.0, scale=0.5, size=(15, 2))
        X = np.vstack([class_0, class_1])
        y = np.array([0.0] * 15 + [1.0] * 15)
        model_before = LogisticRegression()
        model_before.weights = np.zeros(2)
        model_before.bias = 0.0
        cost_before = model_before.compute_cost(X, y)
        model = LogisticRegression(learning_rate=0.1, n_iterations=500)
        model.fit(X, y)
        cost_after = model.compute_cost(X, y)
        self.assertLess(cost_after, cost_before, msg="fit deberia reducir el costo respecto a su valor inicial")

    def test_fit_returns_self(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([0.0, 1.0])
        model = LogisticRegression(learning_rate=0.1, n_iterations=10)
        result = model.fit(X, y)
        self.assertIs(model, result, msg="fit deberia devolver self")


if __name__ == '__main__':
    unittest.main()
