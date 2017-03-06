import unittest
from nonpyrametric import cadlag

class CadlagStepFunctionTest(unittest.TestCase):
    def test_recall(self):
        X = [0,1,4]
        Y = [0, 10, 40]
        f = cadlag.CadlagStepFunction(X=X, Y=Y)
        for x, y in zip(X, Y):
            self.assertEqual(f(x), y)

    def test_interpolation(self):
        X = [0, 1, 2]
        Y = [0, 1, 2]
        f = cadlag.CadlagStepFunction(X, Y)
        self.assertEqual(f(0.5), 0)
        self.assertEqual(f(0.01), 0)
        self.assertEqual(f(0.99), 0)
        self.assertEqual(f(1.5), 1)
        self.assertEqual(f(1.01), 1)
        self.assertEqual(f(1.99), 1)
        self.assertEqual(f(2.5), 2)
        self.assertEqual(f(2.01), 2)
        self.assertEqual(f(2.99), 2)

    def test_bounds(self):
        X = [0, 1, 2]
        Y = [0, 1, 2]
        f = cadlag.CadlagStepFunction(X, Y)
        self.assertEqual(f(5), 2)
        self.assertEqual(f(-1), 0)

    def test_binary_cdf(self):
        X = [0, 1]
        Y = [0.5, 1.0]
        f = cadlag.CadlagStepFunction(X, Y)
        self.assertEqual(f(0), 0.5)
        self.assertEqual(f(0.5), 0.5)
        self.assertEqual(f(1), 1.0)
