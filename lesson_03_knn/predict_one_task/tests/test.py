import unittest
import numpy as np
from main import KNNClassifier


class TestCase(unittest.TestCase):

    def setUp(self):
        self.X_train = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [10.0, 10.0], [10.1, 10.1], [9.9, 10.0]])
        self.y_train = np.array([0, 0, 0, 1, 1, 1])

    def test_predict_near_class_0(self):
        model = KNNClassifier(n_neighbors=3).fit(self.X_train, self.y_train)
        self.assertEqual(0, model._predict_one(np.array([0.05, 0.05])), msg="Un punto cerca del grupo de clase 0 deberia clasificarse como 0")

    def test_predict_near_class_1(self):
        model = KNNClassifier(n_neighbors=3).fit(self.X_train, self.y_train)
        self.assertEqual(1, model._predict_one(np.array([10.05, 10.05])), msg="Un punto cerca del grupo de clase 1 deberia clasificarse como 1")

    def test_predict_exact_training_point(self):
        model = KNNClassifier(n_neighbors=1).fit(self.X_train, self.y_train)
        self.assertEqual(0, model._predict_one(np.array([0.0, 0.0])), msg="Con n_neighbors=1, un punto exacto de entrenamiento deberia devolver su propia etiqueta")


if __name__ == '__main__':
    unittest.main()
