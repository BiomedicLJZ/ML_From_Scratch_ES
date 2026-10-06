import unittest
import numpy as np
from main import AgglomerativeClustering


class TestCase(unittest.TestCase):

    def test_known_cluster_distance(self):
        model = AgglomerativeClustering()
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
        distances = model._pairwise_distances(X)
        self.assertAlmostEqual(9.0, model._cluster_distance([0, 1], [2, 3], distances), places=6,
                                msg="La distancia single-linkage deberia ser la minima distancia entre clusters")

    def test_single_point_clusters(self):
        model = AgglomerativeClustering()
        X = np.array([[0.0, 0.0], [3.0, 4.0]])
        distances = model._pairwise_distances(X)
        self.assertAlmostEqual(5.0, model._cluster_distance([0], [1], distances), places=6,
                                msg="La distancia entre dos clusters de un solo punto deberia ser igual a la distancia entre esos puntos")

    def test_is_symmetric(self):
        model = AgglomerativeClustering()
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
        distances = model._pairwise_distances(X)
        d1 = model._cluster_distance([0, 1], [2, 3], distances)
        d2 = model._cluster_distance([2, 3], [0, 1], distances)
        self.assertAlmostEqual(d1, d2, places=6, msg="La distancia entre clusters deberia ser simetrica")


if __name__ == '__main__':
    unittest.main()
