import unittest
import numpy as np
from main import DecisionTreeClassifier


class TestCase(unittest.TestCase):

    def test_gini_balanced_two_classes(self):
        tree = DecisionTreeClassifier()
        self.assertAlmostEqual(0.5, tree._gini(np.array([0, 0, 1, 1])), places=6, msg="Gini incorrecto para un grupo de 2 clases dividido en partes iguales")

    def test_gini_pure_node(self):
        tree = DecisionTreeClassifier()
        self.assertAlmostEqual(0.0, tree._gini(np.array([0, 0, 0, 0])), places=6, msg="Gini deberia ser 0 para un nodo puro")

    def test_gini_unbalanced(self):
        tree = DecisionTreeClassifier()
        self.assertAlmostEqual(0.375, tree._gini(np.array([0, 0, 0, 1])), places=6, msg="Gini incorrecto para una division 3 a 1")


if __name__ == '__main__':
    unittest.main()
