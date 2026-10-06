import unittest
import numpy as np
from main import LogisticRegression


class TestCase(unittest.TestCase):

    def test_gradients_known_values(self):
        model = LogisticRegression()
        model.weights = np.array([0.0])
        model.bias = 0.0
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0.0, 1.0, 1.0])
        dw, db = model.compute_gradients(X, y)
        np.testing.assert_allclose(dw, np.array([-2 / 3]), atol=1e-8, err_msg="dw incorrecto")
        self.assertAlmostEqual(-1 / 6, db, places=6, msg="db incorrecto")

    def test_gradients_shape(self):
        model = LogisticRegression()
        model.weights = np.array([1.0, 1.0, 1.0])
        model.bias = 0.0
        X = np.ones((5, 3))
        y = np.zeros(5)
        dw, db = model.compute_gradients(X, y)
        self.assertEqual((3,), dw.shape, msg="dw deberia tener la misma forma que weights")
        self.assertIsInstance(db, (float, np.floating), msg="db deberia ser un escalar")

    def test_gradients_zero_when_predictions_match_labels(self):
        model = LogisticRegression()
        model.weights = np.array([1000.0])
        model.bias = 0.0
        X = np.array([[1.0], [-1.0]])
        y = np.array([1.0, 0.0])
        dw, db = model.compute_gradients(X, y)
        np.testing.assert_allclose(dw, np.zeros_like(dw), atol=1e-6, err_msg="dw deberia ser ~0 cuando las predicciones ya coinciden con las etiquetas")
        self.assertAlmostEqual(0.0, db, places=5, msg="db deberia ser ~0 cuando las predicciones ya coinciden con las etiquetas")


if __name__ == '__main__':
    unittest.main()
