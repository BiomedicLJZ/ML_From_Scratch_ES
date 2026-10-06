import unittest
import numpy as np
from main import LinearRegression


class TestCase(unittest.TestCase):

    def test_gradients_known_values(self):
        model = LinearRegression()
        model.weights = np.array([1.0])
        model.bias = 0.0
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 3.0, 5.0])
        dw, db = model.compute_gradients(X, y)
        np.testing.assert_allclose(dw, np.array([-6.0]), err_msg="dw incorrecto")
        self.assertAlmostEqual(-8 / 3, db, places=6, msg="db incorrecto")

    def test_gradients_zero_at_perfect_fit(self):
        model = LinearRegression()
        model.weights = np.array([2.0])
        model.bias = 1.0
        X = np.array([[1.0], [2.0], [3.0]])
        y = 2 * X.flatten() + 1
        dw, db = model.compute_gradients(X, y)
        np.testing.assert_allclose(dw, np.zeros_like(dw), atol=1e-8, err_msg="dw deberia ser cero en un ajuste perfecto")
        self.assertAlmostEqual(0.0, db, places=6, msg="db deberia ser cero en un ajuste perfecto")

    def test_gradients_shape(self):
        model = LinearRegression()
        model.weights = np.array([1.0, 1.0, 1.0])
        model.bias = 0.0
        X = np.ones((4, 3))
        y = np.zeros(4)
        dw, db = model.compute_gradients(X, y)
        self.assertEqual((3,), dw.shape, msg="dw deberia tener la misma forma que weights")


if __name__ == '__main__':
    unittest.main()
