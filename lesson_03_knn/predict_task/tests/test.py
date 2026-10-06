import unittest
import numpy as np
from main import KNNClassifier


class TestCase(unittest.TestCase):

    def setUp(self):
        self.X_train = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [10.0, 10.0], [10.1, 10.1], [9.9, 10.0]])
        self.y_train = np.array([0, 0, 0, 1, 1, 1])

    def test_predict_batch(self):
        model = KNNClassifier(n_neighbors=3).fit(self.X_train, self.y_train)
        X_test = np.array([[0.05, 0.05], [10.05, 10.05]])
        np.testing.assert_array_equal(model.predict(X_test), np.array([0, 1]), err_msg="Predicciones de lote incorrectas")

    def test_predict_returns_array_of_correct_length(self):
        model = KNNClassifier(n_neighbors=3).fit(self.X_train, self.y_train)
        X_test = np.array([[0.0, 0.0], [10.0, 10.0], [0.1, 0.0]])
        result = model.predict(X_test)
        self.assertEqual(3, len(result), msg="predict deberia devolver una etiqueta por cada fila de entrada")

    def test_predict_all_training_points_recovers_labels(self):
        model = KNNClassifier(n_neighbors=1).fit(self.X_train, self.y_train)
        np.testing.assert_array_equal(model.predict(self.X_train), self.y_train, err_msg="Con n_neighbors=1, predecir sobre el conjunto de entrenamiento deberia recuperar exactamente las etiquetas de entrenamiento")


if __name__ == '__main__':
    unittest.main()
