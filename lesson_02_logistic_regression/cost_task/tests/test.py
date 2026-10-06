import unittest
import numpy as np
from main import LogisticRegression


class TestCase(unittest.TestCase):

    def test_cost_low_when_confident_and_correct(self):
        model = LogisticRegression()
        model.weights = np.array([10.0])
        model.bias = 0.0
        X = np.array([[10.0], [-10.0]])
        y = np.array([1.0, 0.0])
        cost = model.compute_cost(X, y)
        self.assertLess(cost, 0.01, msg="El costo deberia ser cercano a 0 para predicciones confiadas y correctas")

    def test_cost_high_when_confident_and_wrong(self):
        model = LogisticRegression()
        model.weights = np.array([10.0])
        model.bias = 0.0
        X = np.array([[10.0], [-10.0]])
        y = np.array([0.0, 1.0])
        cost = model.compute_cost(X, y)
        self.assertGreater(cost, 10.0, msg="El costo deberia ser grande para predicciones confiadas y equivocadas")

    def test_cost_no_nan_or_inf(self):
        model = LogisticRegression()
        model.weights = np.array([50.0])
        model.bias = 0.0
        X = np.array([[50.0], [-50.0]])
        y = np.array([0.0, 1.0])
        cost = model.compute_cost(X, y)
        self.assertTrue(np.isfinite(cost), msg="El costo debe ser finito incluso para predicciones extremadamente confiadas (recorta p primero)")


if __name__ == '__main__':
    unittest.main()
