import unittest
import numpy as np
from main import DBSCAN


class TestCase(unittest.TestCase):

    def test_fit_separates_two_blobs_and_flags_outlier(self):
        blob_a = np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [0.2, 0.2]])
        blob_b = np.array([[10.0, 10.0], [10.2, 10.0], [10.0, 10.2], [10.2, 10.2]])
        outlier = np.array([[50.0, 50.0]])
        X = np.vstack([blob_a, blob_b, outlier])
        model = DBSCAN(eps=0.5, min_samples=3)
        model.fit(X)
        labels = model.labels_
        self.assertEqual(1, len(set(labels[0:4].tolist())), msg="Todo el grupo A deberia compartir una sola etiqueta de cluster")
        self.assertEqual(1, len(set(labels[4:8].tolist())), msg="Todo el grupo B deberia compartir una sola etiqueta de cluster")
        self.assertNotEqual(labels[0], labels[4], msg="Los dos grupos deberian obtener etiquetas de cluster distintas")
        self.assertNotIn(-1, labels[0:8], msg="Ningun punto dentro de un grupo denso deberia ser ruido")
        self.assertEqual(-1, labels[8], msg="El punto aislado y lejano deberia etiquetarse como ruido")

    def test_fit_all_noise_when_too_sparse(self):
        X = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]])
        model = DBSCAN(eps=0.5, min_samples=2)
        model.fit(X)
        np.testing.assert_array_equal(model.labels_, np.array([-1, -1, -1]), err_msg="Cada punto esta aislado, asi que todos deberian ser ruido")

    def test_fit_single_dense_cluster(self):
        rng_points = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.1, -0.1], [0.0, 0.1]])
        model = DBSCAN(eps=0.5, min_samples=3)
        model.fit(rng_points)
        self.assertTrue(np.all(model.labels_ == model.labels_[0]), msg="Un solo grupo denso deberia terminar como un unico cluster")
        self.assertNotEqual(-1, model.labels_[0], msg="Un grupo denso no deberia etiquetarse como ruido")


if __name__ == '__main__':
    unittest.main()
