import unittest
import numpy as np
from main import KMeans


class TestCase(unittest.TestCase):

    def test_fit_separates_two_blobs(self):
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [10.0, 10.0], [10.0, 11.0], [11.0, 10.0]])
        initial_centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
        model = KMeans(n_clusters=2, n_iterations=10)
        model.fit(X, initial_centroids)
        labels = model.labels_
        self.assertEqual(labels[0], labels[1], msg="Los puntos del mismo grupo deberian compartir etiqueta de cluster")
        self.assertEqual(labels[1], labels[2], msg="Los puntos del mismo grupo deberian compartir etiqueta de cluster")
        self.assertEqual(labels[3], labels[4], msg="Los puntos del mismo grupo deberian compartir etiqueta de cluster")
        self.assertEqual(labels[4], labels[5], msg="Los puntos del mismo grupo deberian compartir etiqueta de cluster")
        self.assertNotEqual(labels[0], labels[3], msg="Los puntos de grupos distintos deberian tener etiquetas distintas")

    def test_fit_centroids_close_to_blob_means(self):
        X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [10.0, 10.0], [10.0, 11.0], [11.0, 10.0]])
        initial_centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
        model = KMeans(n_clusters=2, n_iterations=10)
        model.fit(X, initial_centroids)
        label_of_first_blob = model.labels_[0]
        centroid_first_blob = model.centroids_[label_of_first_blob]
        expected_first_blob_mean = X[:3].mean(axis=0)
        np.testing.assert_allclose(centroid_first_blob, expected_first_blob_mean, atol=1e-8, err_msg="El centroide deberia converger a la media de sus puntos asignados")

    def test_fit_returns_self(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        model = KMeans(n_clusters=2, n_iterations=1)
        result = model.fit(X, np.array([[0.0, 0.0], [1.0, 1.0]]))
        self.assertIs(model, result, msg="fit deberia devolver self")


if __name__ == '__main__':
    unittest.main()
