import unittest
import numpy as np
from main import DBSCAN


class TestCase(unittest.TestCase):

    def test_expand_cluster_labels_whole_blob(self):
        X = np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [0.2, 0.2], [10.0, 10.0]])
        model = DBSCAN(eps=0.5, min_samples=3)
        labels = np.full(5, -1, dtype=int)
        neighbors = model._region_query(X, 0)
        model._expand_cluster(X, labels, 0, neighbors, 0)
        np.testing.assert_array_equal(labels, np.array([0, 0, 0, 0, -1]), err_msg="La expansion deberia etiquetar todo el grupo denso y no tocar el punto lejano")

    def test_expand_cluster_does_not_revisit_labeled_points(self):
        X = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
        model = DBSCAN(eps=1.0, min_samples=2)
        labels = np.array([-1, 5, -1])
        neighbors = model._region_query(X, 0)
        model._expand_cluster(X, labels, 0, neighbors, 0)
        self.assertEqual(5, labels[1], msg="Un punto que ya tiene una etiqueta de cluster no deberia sobrescribirse")
        self.assertEqual(0, labels[0], msg="El punto inicial deberia recibir cluster_id")

    def test_expand_cluster_does_not_grow_past_sparse_points(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        model = DBSCAN(eps=1.5, min_samples=3)
        labels = np.full(3, -1, dtype=int)
        neighbors = model._region_query(X, 1)
        model._expand_cluster(X, labels, 1, neighbors, 7)
        np.testing.assert_array_equal(labels, np.array([7, 7, 7]), err_msg="Los tres puntos son vecinos mutuos aqui y deberian compartir la etiqueta de cluster")


if __name__ == '__main__':
    unittest.main()
