import unittest
import numpy as np
from main import PCA


class TestCase(unittest.TestCase):

    def test_center_basic(self):
        pca = PCA()
        pca.mean_ = np.array([1.0, 2.0])
        X = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.0]])
        expected = np.array([[0.0, 0.0], [2.0, 2.0], [-2.0, -2.0]])
        np.testing.assert_allclose(pca._center(X), expected, err_msg="Salida centrada incorrecta")

    def test_center_zero_mean(self):
        pca = PCA()
        pca.mean_ = np.zeros(3)
        X = np.array([[1.0, -1.0, 2.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(pca._center(X), X, err_msg="Centrar con media cero no deberia cambiar X")

    def test_center_does_not_mutate_input(self):
        pca = PCA()
        pca.mean_ = np.array([1.0])
        X = np.array([[5.0], [10.0]])
        X_copy = X.copy()
        pca._center(X)
        np.testing.assert_allclose(X, X_copy, err_msg="_center no deberia mutar su array de entrada")


if __name__ == '__main__':
    unittest.main()
