import unittest
import numpy as np
from main import DecisionTreeClassifier


class TestCase(unittest.TestCase):

    def test_split_single_feature(self):
        tree = DecisionTreeClassifier()
        X = np.array([[1.0], [2.0], [8.0], [9.0]])
        y = np.array([0, 0, 1, 1])
        X_left, y_left, X_right, y_right = tree._split_dataset(X, y, 0, 5.0)
        np.testing.assert_array_equal(X_left, np.array([[1.0], [2.0]]), err_msg="Particion izquierda de X incorrecta")
        np.testing.assert_array_equal(y_left, np.array([0, 0]), err_msg="Particion izquierda de y incorrecta")
        np.testing.assert_array_equal(X_right, np.array([[8.0], [9.0]]), err_msg="Particion derecha de X incorrecta")
        np.testing.assert_array_equal(y_right, np.array([1, 1]), err_msg="Particion derecha de y incorrecta")

    def test_split_boundary_is_inclusive_left(self):
        tree = DecisionTreeClassifier()
        X = np.array([[1.0], [5.0], [9.0]])
        y = np.array([0, 1, 2])
        X_left, y_left, X_right, y_right = tree._split_dataset(X, y, 0, 5.0)
        self.assertEqual(2, len(y_left), msg="Una fila igual al umbral deberia ir a la izquierda")
        self.assertEqual(1, len(y_right), msg="Numero incorrecto de filas enviadas a la derecha")

    def test_split_second_feature(self):
        tree = DecisionTreeClassifier()
        X = np.array([[0.0, 3.0], [0.0, 7.0]])
        y = np.array([0, 1])
        X_left, y_left, X_right, y_right = tree._split_dataset(X, y, 1, 5.0)
        np.testing.assert_array_equal(y_left, np.array([0]), err_msg="Division incorrecta al usar el indice de caracteristica 1")
        np.testing.assert_array_equal(y_right, np.array([1]), err_msg="Division incorrecta al usar el indice de caracteristica 1")


if __name__ == '__main__':
    unittest.main()
