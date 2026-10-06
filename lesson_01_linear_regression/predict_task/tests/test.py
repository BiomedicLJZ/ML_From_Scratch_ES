import unittest
import numpy as np
from main import LinearRegression


class TestCase(unittest.TestCase):

    def test_predict_2d(self):
        model = LinearRegression()
        model.weights = np.array([2.0, -1.0])
        model.bias = 0.5
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.array([0.5, 2.5])
        np.testing.assert_allclose(model.predict(X), expected, err_msg="Predicciones incorrectas para una entrada con 2 caracteristicas")

    def test_predict_1d_feature(self):
        model = LinearRegression()
        model.weights = np.array([3.0])
        model.bias = -2.0
        X = np.array([[0.0], [1.0], [2.0]])
        expected = np.array([-2.0, 1.0, 4.0])
        np.testing.assert_allclose(model.predict(X), expected, err_msg="Predicciones incorrectas para una entrada con una sola caracteristica")

    def test_predict_zero_weights(self):
        model = LinearRegression()
        model.weights = np.zeros(3)
        model.bias = 7.0
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected = np.array([7.0, 7.0])
        np.testing.assert_allclose(model.predict(X), expected, err_msg="Predicciones incorrectas cuando los pesos son cero")


if __name__ == '__main__':
    unittest.main()
