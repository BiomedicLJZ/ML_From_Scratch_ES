import unittest
import numpy as np
from main import DecisionTreeClassifier


class TestCase(unittest.TestCase):

    def test_perfect_training_accuracy_when_separable(self):
        rng = np.random.default_rng(0)
        blob_a = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(6, 2))
        blob_b = rng.normal(loc=[10.0, 10.0], scale=0.3, size=(6, 2))
        X = np.vstack([blob_a, blob_b])
        y = np.array([0] * 6 + [1] * 6)
        tree = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
        tree.fit(X, y)
        predictions = tree.predict(X)
        self.assertEqual(1.0, np.mean(predictions == y), msg="El arbol deberia ajustar perfectamente un conjunto de entrenamiento claramente separable")

    def test_root_is_leaf_when_classes_already_pure(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1, 1, 1])
        tree = DecisionTreeClassifier()
        tree.fit(X, y)
        self.assertTrue(tree.tree_['leaf'], msg="Un conjunto de datos con una sola clase deberia construir un unico nodo hoja")
        self.assertEqual(1, tree.tree_['class'], msg="La clase de la hoja deberia coincidir con la unica clase presente")

    def test_max_depth_limits_splitting(self):
        rng = np.random.default_rng(1)
        blob_a = rng.normal(loc=[0.0, 0.0], scale=0.3, size=(6, 2))
        blob_b = rng.normal(loc=[10.0, 10.0], scale=0.3, size=(6, 2))
        X = np.vstack([blob_a, blob_b])
        y = np.array([0] * 6 + [1] * 6)
        tree = DecisionTreeClassifier(max_depth=1, min_samples_split=2)
        tree.fit(X, y)
        self.assertFalse(tree.tree_['leaf'], msg="Con max_depth=1 la raiz deberia seguir dividiendo estos datos separables")
        self.assertTrue(tree.tree_['left']['leaf'], msg="En profundidad 1 con max_depth=1, los hijos deben ser hojas")
        self.assertTrue(tree.tree_['right']['leaf'], msg="En profundidad 1 con max_depth=1, los hijos deben ser hojas")


if __name__ == '__main__':
    unittest.main()
