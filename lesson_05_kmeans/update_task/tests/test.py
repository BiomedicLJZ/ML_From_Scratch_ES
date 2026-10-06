import unittest
import numpy as np
from main import KMeans


class TestCase(unittest.TestCase):

    def test_update_two_clusters(self):
        model = KMeans()
        X = np.array([[0.0, 0.0], [0.2, 0.0], [10.0, 10.0], [10.2, 10.0]])
        labels = np.array([0, 0, 1, 1])
        centroids = model._update_centroids(X, labels, 2)
        expected = np.array([[0.1, 0.0], [10.1, 10.0]])
        np.testing.assert_allclose(centroids, expected, err_msg="Centroides recalculados incorrectos")

    def test_update_single_point_clusters(self):
        model = KMeans()
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([0, 1, 2])
        centroids = model._update_centroids(X, labels, 3)
        np.testing.assert_allclose(centroids, X, err_msg="Un cluster con un solo punto deberia tener ese punto como centroide")

    def test_update_output_shape(self):
        model = KMeans()
        X = np.zeros((6, 4))
        labels = np.array([0, 0, 0, 1, 1, 1])
        centroids = model._update_centroids(X, labels, 2)
        self.assertEqual((2, 4), centroids.shape, msg="centroids deberia tener forma (n_clusters, n_features)")


if __name__ == '__main__':
    unittest.main()
