import unittest
import numpy as np
from main import KMeans


class TestCase(unittest.TestCase):

    def test_assign_two_obvious_clusters(self):
        model = KMeans()
        X = np.array([[0.0, 0.0], [0.2, 0.0], [10.0, 10.0], [10.2, 10.0]])
        centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels = model._assign_clusters(X, centroids)
        np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1]), err_msg="Los puntos deberian asignarse a su centroide mas cercano")

    def test_assign_exact_centroid_match(self):
        model = KMeans()
        X = np.array([[1.0, 1.0], [5.0, 5.0]])
        centroids = np.array([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]])
        labels = model._assign_clusters(X, centroids)
        np.testing.assert_array_equal(labels, np.array([0, 1]), err_msg="Un punto exactamente sobre un centroide deberia asignarse a el")

    def test_assign_output_shape(self):
        model = KMeans()
        X = np.zeros((7, 3))
        centroids = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        labels = model._assign_clusters(X, centroids)
        self.assertEqual((7,), labels.shape, msg="labels deberia tener una entrada por muestra")


if __name__ == '__main__':
    unittest.main()
