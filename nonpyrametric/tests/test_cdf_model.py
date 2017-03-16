import unittest
from nonpyrametric.cdf_model import CDFModel
import numpy as np
from scipy.stats import norm

class CDFModelTest(unittest.TestCase):
    def test_bernoulli_data(self):
        X = [0, 0, 1, 1]
        m = CDFModel()
        m.fit(X)
        self.assertEqual(m.cdf(0), 0.5)
        self.assertEqual(m.cdf(0.5), 0.5)
        self.assertEqual(m.cdf(1), 1.0)

    def test_ci_coverage_normal(self):
        normal_rvs = np.random.normal(0, 1, 50)
        m = CDFModel()
        m.fit(normal_rvs)
        X = np.linspace(-2, 2)
        for x in X:
            true_val = norm.cdf(x, 0, 1)
            low, high = m.cdf_ci(x, alpha=0.01)
            self.assertLess(low, true_val)
            self.assertGreaterEqual(low, 0)
            self.assertGreater(high, true_val)
            self.assertLessEqual(high, 1)

    def test_ci_mean_bernoulli(self):
        X = [0]*100 + [1]*100
        m = CDFModel()
        m.fit(X)
        self.assertEqual(m.mean(), 0.5)
        low, high = m.mean_ci(0.05)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
