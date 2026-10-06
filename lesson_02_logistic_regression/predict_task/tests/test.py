import unittest
import numpy as np
from main import LogisticRegression


class TestCase(unittest.TestCase):

    def test_predict_default_threshold(self):
        model = LogisticRegression()
        model.weights = np.array([1.0, -1.0])
        model.bias = 0.0
        X = np.array([[2.0, 2.0], [0.0, 5.0]])
        np.testing.assert_array_equal(model.predict(X), np.array([1, 0]), err_msg="Etiquetas de clase incorrectas con el umbral por defecto")

    def test_predict_returns_int_dtype(self):
        model = LogisticRegression()
        model.weights = np.array([1.0])
        model.bias = 0.0
        X = np.array([[1.0], [-1.0]])
        result = model.predict(X)
        self.assertTrue(np.issubdtype(result.dtype, np.integer), msg="predict deberia devolver etiquetas enteras")

    def test_predict_custom_threshold(self):
        model = LogisticRegression()
        model.weights = np.array([1.0, -1.0])
        model.bias = 0.0
        X = np.array([[2.0, 2.0]])
        low_threshold_pred = model.predict(X, threshold=0.1)
        high_threshold_pred = model.predict(X, threshold=0.9)
        self.assertEqual(1, low_threshold_pred[0], msg="Un umbral bajo deberia clasificar una muestra con p=0.5 como 1")
        self.assertEqual(0, high_threshold_pred[0], msg="Un umbral alto deberia clasificar una muestra con p=0.5 como 0")


if __name__ == '__main__':
    unittest.main()
