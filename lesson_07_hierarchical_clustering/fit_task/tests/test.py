import unittest
import numpy as np
from main import AgglomerativeClustering


class TestCase(unittest.TestCase):

    def test_two_obvious_pairs(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(X)
        labels = model.labels_
        self.assertEqual(labels[0], labels[1], msg="Los puntos 0 y 1 deberian terminar en el mismo cluster")
        self.assertEqual(labels[2], labels[3], msg="Los puntos 2 y 3 deberian terminar en el mismo cluster")
        self.assertNotEqual(labels[0], labels[2], msg="Los dos pares deberian estar en clusters distintos")

    def test_respects_n_clusters(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0], [50.0, 0.0], [51.0, 0.0]])
        model = AgglomerativeClustering(n_clusters=3)
        model.fit(X)
        self.assertEqual(3, len(set(model.labels_.tolist())), msg="Deberia terminar con exactamente n_clusters etiquetas distintas")

    def test_labels_length_matches_samples(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(X)
        self.assertEqual(X.shape[0], len(model.labels_), msg="Deberia producir una etiqueta por muestra")


if __name__ == '__main__':
    unittest.main()
