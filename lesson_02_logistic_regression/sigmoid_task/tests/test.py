import unittest
import numpy as np
from main import LogisticRegression


class TestCase(unittest.TestCase):

    def test_sigmoid_at_zero(self):
        model = LogisticRegression()
        self.assertAlmostEqual(0.5, model._sigmoid(0), places=6, msg="sigmoid(0) deberia ser 0.5")

    def test_sigmoid_saturates(self):
        model = LogisticRegression()
        result = model._sigmoid(np.array([-100.0, 100.0]))
        np.testing.assert_allclose(result, np.array([0.0, 1.0]), atol=1e-6, err_msg="sigmoid deberia saturarse cerca de 0 y 1")

    def test_sigmoid_is_monotonic(self):
        model = LogisticRegression()
        zs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        values = model._sigmoid(zs)
        self.assertTrue(np.all(np.diff(values) > 0), msg="sigmoid deberia ser estrictamente creciente")


if __name__ == '__main__':
    unittest.main()
