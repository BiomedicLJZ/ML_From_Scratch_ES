import unittest
import numpy as np
from main import PCA


class TestCase(unittest.TestCase):

    def _make_data(self):
        rng = np.random.default_rng(0)
        t = np.linspace(-5, 5, 50)
        noise = rng.normal(0, 0.01, size=50)
        return np.column_stack([t, 0.3 * t + noise])

    def test_components_shape(self):
        X = self._make_data()
        pca = PCA(n_components=1)
        pca.fit(X)
        self.assertEqual((1, 2), pca.components_.shape, msg="components_ deberia tener forma (n_components, n_features)")
        self.assertEqual((1,), pca.explained_variance_.shape, msg="explained_variance_ deberia tener forma (n_components,)")

    def test_components_direction_sign_agnostic(self):
        X = self._make_data()
        pca = PCA(n_components=1)
        pca.fit(X)
        expected_dir = np.array([1.0, 0.3])
        expected_dir = expected_dir / np.linalg.norm(expected_dir)
        alignment = abs(np.dot(pca.components_[0], expected_dir))
        self.assertGreater(alignment, 0.99, msg="El primer componente deberia alinearse (salvo el signo) con la direccion principal de varianza de los datos")

    def test_explained_variance_matches_top_eigenvalue(self):
        X = self._make_data()
        X_centered = X - X.mean(axis=0)
        expected_eigenvalues = np.sort(np.linalg.eigvalsh(np.cov(X_centered, rowvar=False)))[::-1]
        pca = PCA(n_components=1)
        pca.fit(X)
        self.assertAlmostEqual(expected_eigenvalues[0], pca.explained_variance_[0], places=6, msg="explained_variance_ deberia coincidir con el mayor autovalor de la matriz de covarianza")

    def test_two_components_are_orthogonal(self):
        X = self._make_data()
        pca = PCA(n_components=2)
        pca.fit(X)
        dot = np.dot(pca.components_[0], pca.components_[1])
        self.assertAlmostEqual(0.0, dot, places=6, msg="Los componentes principales deberian ser ortogonales")


if __name__ == '__main__':
    unittest.main()
