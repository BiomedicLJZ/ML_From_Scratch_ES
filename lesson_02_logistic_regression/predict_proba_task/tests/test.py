import unittest
import numpy as np
from main import LogisticRegression


class TestCase(unittest.TestCase):

    def test_predict_proba_known_values(self):
        model = LogisticRegression()
        model.weights = np.array([1.0, -1.0])
        model.bias = 0.0
        X = np.array([[2.0, 2.0], [0.0, 5.0]])
        expected = np.array([0.5, 0.00669285])
        np.testing.assert_allclose(model.predict_proba(X), expected, atol=1e-6, err_msg="Probabilidades incorrectas")

    def test_predict_proba_range(self):
        model = LogisticRegression()
        model.weights = np.array([3.0, -2.0, 1.0])
        model.bias = 0.5
        rng = np.random.default_rng(1)
        X = rng.uniform(-10, 10, size=(20, 3))
        probs = model.predict_proba(X)
        self.assertTrue(np.all((probs > 0) & (probs < 1)), msg="Las probabilidades deben estar estrictamente entre 0 y 1")

    def test_predict_proba_shape(self):
        model = LogisticRegression()
        model.weights = np.zeros(4)
        model.bias = 0.0
        X = np.zeros((7, 4))
        self.assertEqual((7,), model.predict_proba(X).shape, msg="predict_proba deberia devolver un valor por muestra")


if __name__ == '__main__':
    unittest.main()
