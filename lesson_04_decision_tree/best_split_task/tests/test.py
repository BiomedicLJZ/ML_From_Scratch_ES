import unittest
import numpy as np
from main import DecisionTreeClassifier


class TestCase(unittest.TestCase):

    def test_best_split_perfectly_separable(self):
        tree = DecisionTreeClassifier()
        X = np.array([[1.0], [2.0], [8.0], [9.0]])
        y = np.array([0, 0, 1, 1])
        feature, threshold = tree._best_split(X, y)
        self.assertEqual(0, feature, msg="Con una sola caracteristica, la mejor division debe usar la caracteristica 0")
        self.assertGreaterEqual(threshold, 2.0, msg="El umbral deberia ser al menos el mayor valor de la clase 0")
        self.assertLess(threshold, 8.0, msg="El umbral deberia ser menor que el menor valor de la clase 1")
        X_left, y_left, X_right, y_right = tree._split_dataset(X, y, feature, threshold)
        self.assertAlmostEqual(0.0, tree._gini(y_left), places=6, msg="La particion izquierda deberia ser pura")
        self.assertAlmostEqual(0.0, tree._gini(y_right), places=6, msg="La particion derecha deberia ser pura")

    def test_best_split_picks_informative_feature(self):
        tree = DecisionTreeClassifier()
        # la caracteristica 0 es puro ruido (mismo valor para ambas clases), la caracteristica 1 separa perfectamente
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 8.0], [5.0, 9.0]])
        y = np.array([0, 0, 1, 1])
        feature, threshold = tree._best_split(X, y)
        self.assertEqual(1, feature, msg="La division deberia usar la caracteristica informativa (indice 1)")

    def test_best_split_returns_none_when_no_split_helps(self):
        tree = DecisionTreeClassifier()
        X = np.array([[5.0], [5.0], [5.0], [5.0]])
        y = np.array([0, 1, 0, 1])
        feature, threshold = tree._best_split(X, y)
        self.assertIsNone(feature, msg="Ninguna caracteristica puede dividir filas identicas, feature deberia ser None")
        self.assertIsNone(threshold, msg="Ninguna caracteristica puede dividir filas identicas, threshold deberia ser None")


if __name__ == '__main__':
    unittest.main()
