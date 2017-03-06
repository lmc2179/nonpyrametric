import unittest
from nonpyrametric.cdf_model import CDFModel

class CDFModelTest(unittest.TestCase):
    def test_bernoulli_data(self):
        X = [0, 0, 1, 1]
        m = CDFModel()
        m.fit(X)
        self.assertEqual(m.cdf(0), 0.5)
        self.assertEqual(m.cdf(0.5), 0.5)
        self.assertEqual(m.cdf(1), 1.0)