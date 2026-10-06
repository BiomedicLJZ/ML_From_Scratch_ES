import unittest
import numpy as np
from main import DBSCAN


class TestCase(unittest.TestCase):

    def test_region_query_finds_close_points(self):
        X = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 5.0]])
        model = DBSCAN(eps=0.5, min_samples=2)
        result = model._region_query(X, 0)
        np.testing.assert_array_equal(np.sort(result), np.array([0, 1]), err_msg="Vecindario incorrecto para el punto 0")

    def test_region_query_includes_self(self):
        X = np.array([[1.0, 1.0], [100.0, 100.0]])
        model = DBSCAN(eps=0.1, min_samples=1)
        result = model._region_query(X, 1)
        np.testing.assert_array_equal(result, np.array([1]), err_msg="El indice de un punto siempre deberia estar incluido")

    def test_region_query_respects_eps(self):
        X = np.array([[0.0], [1.0], [2.0], [10.0]])
        model = DBSCAN(eps=1.5, min_samples=1)
        result = model._region_query(X, 1)
        np.testing.assert_array_equal(np.sort(result), np.array([0, 1, 2]), err_msg="Manejo incorrecto del radio del vecindario")


if __name__ == '__main__':
    unittest.main()
