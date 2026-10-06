import unittest
import numpy as np
from main import LinearRegression


class TestCase(unittest.TestCase):

    def test_cost_nonzero_error(self):
        model = LinearRegression()
        model.weights = np.array([2.0, -1.0])
        model.bias = 0.5
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 5.0])
        self.assertAlmostEqual(3.25, model.compute_cost(X, y), places=6, msg="MSE incorrecto para un lote pequeno de 2 muestras")

    def test_cost_zero_when_perfect(self):
        model = LinearRegression()
        model.weights = np.array([1.0])
        model.bias = 0.0
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(0.0, model.compute_cost(X, y), places=6, msg="El MSE deberia ser 0 para predicciones perfectas")

    def test_cost_is_scalar(self):
        model = LinearRegression()
        model.weights = np.array([0.5, 0.5])
        model.bias = 1.0
        X = np.array([[2.0, 2.0], [4.0, 0.0]])
        y = np.array([0.0, 0.0])
        result = model.compute_cost(X, y)
        self.assertIsInstance(result, (float, np.floating), msg="compute_cost deberia devolver un escalar, no un array")


if __name__ == '__main__':
    unittest.main()
