import unittest
import numpy as np
from main import PCA


class TestCase(unittest.TestCase):

    def _make_data(self):
        rng = np.random.default_rng(0)
        t = np.linspace(-5, 5, 50)
        noise = rng.normal(0, 0.01, size=50)
        return np.column_stack([t, 0.3 * t + noise])

    def test_transform_output_shape(self):
        X = self._make_data()
        pca = PCA(n_components=1)
        pca.fit(X)
        reduced = pca.transform(X)
        self.assertEqual((50, 1), reduced.shape, msg="transform deberia devolver (n_samples, n_components)")

    def test_mean_maps_to_origin(self):
        X = self._make_data()
        pca = PCA(n_components=1)
        pca.fit(X)
        projected_mean = pca.transform(pca.mean_.reshape(1, -1))
        np.testing.assert_allclose(projected_mean, np.zeros((1, 1)), atol=1e-8, err_msg="La media de entrenamiento deberia proyectarse al origen")

    def test_fit_transform_matches_fit_then_transform(self):
        X = self._make_data()
        pca_a = PCA(n_components=1)
        combined = pca_a.fit_transform(X)
        pca_b = PCA(n_components=1)
        pca_b.fit(X)
        separate = pca_b.transform(X)
        np.testing.assert_allclose(combined, separate, err_msg="fit_transform deberia coincidir con llamar fit y luego transform")

    def test_transform_preserves_relative_distances_along_axis(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        pca = PCA(n_components=1)
        reduced = pca.fit_transform(X)
        diffs = np.abs(np.diff(reduced.flatten()))
        np.testing.assert_allclose(diffs, np.array([1.0, 1.0]), err_msg="Puntos colineales equiespaciados deberian seguir equiespaciados tras la proyeccion")


if __name__ == '__main__':
    unittest.main()
