import unittest
import numpy as np
from main import AgglomerativeClustering


class TestCase(unittest.TestCase):

    def test_known_distances(self):
        model = AgglomerativeClustering()
        X = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]])
        expected = np.array([
            [0.0, 5.0, 3.0],
            [5.0, 0.0, 4.0],
            [3.0, 4.0, 0.0],
        ])
        np.testing.assert_allclose(model._pairwise_distances(X), expected, err_msg="Matriz de distancias por pares incorrecta")

    def test_shape(self):
        model = AgglomerativeClustering()
        X = np.zeros((5, 3))
        self.assertEqual((5, 5), model._pairwise_distances(X).shape, msg="La matriz de distancias deberia ser (n_samples, n_samples)")

    def test_symmetric_and_zero_diagonal(self):
        model = AgglomerativeClustering()
        rng = np.random.default_rng(1)
        X = rng.uniform(-5, 5, size=(6, 2))
        distances = model._pairwise_distances(X)
        np.testing.assert_allclose(distances, distances.T, err_msg="La matriz de distancias deberia ser simetrica")
        np.testing.assert_allclose(np.diag(distances), np.zeros(6), err_msg="La diagonal deberia ser toda ceros")


if __name__ == '__main__':
    unittest.main()
