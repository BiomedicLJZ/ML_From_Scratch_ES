import unittest
import numpy as np
from main import KNNClassifier


class TestCase(unittest.TestCase):

    def test_known_3_4_5_triangle(self):
        model = KNNClassifier()
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        self.assertAlmostEqual(5.0, model._euclidean_distance(a, b), places=6, msg="Distancia incorrecta para un triangulo 3-4-5")

    def test_zero_distance(self):
        model = KNNClassifier()
        a = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(0.0, model._euclidean_distance(a, a), places=6, msg="La distancia de un punto a si mismo deberia ser 0")

    def test_single_axis(self):
        model = KNNClassifier()
        a = np.array([0.0])
        b = np.array([7.0])
        self.assertAlmostEqual(7.0, model._euclidean_distance(a, b), places=6, msg="Distancia 1-D incorrecta")


if __name__ == '__main__':
    unittest.main()
