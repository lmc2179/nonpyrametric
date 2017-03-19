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
        X = [0]*100 + [1]*400
        TRUE_MEAN = 0.8
        m = CDFModel()
        m.fit(X)
        self.assertEqual(m.mean(), TRUE_MEAN)
        low, high = m.mean_ci(0.05)
        self.assertLess(low, TRUE_MEAN)
        self.assertGreater(high, TRUE_MEAN)

    def test_ci_mean_multinoulli(self):
        X = [0]*100 + [1]*300 + [2]*100
        TRUE_MEAN = 1*.6 + 2*.2
        m = CDFModel()
        m.fit(X)
        self.assertEqual(m.mean(), TRUE_MEAN)
        low, high = m.mean_ci(0.05)
        self.assertLess(low, TRUE_MEAN)
        self.assertGreater(high, TRUE_MEAN)

    def test_ci_mean_uniform_nonnegative(self):
        X = np.random.uniform(0, 1, 100)
        TRUE_MEAN = 0.5
        EMPIRICAL_MEAN = np.mean(X)
        m = CDFModel()
        m.fit(X)
        self.assertAlmostEqual(m.mean(), EMPIRICAL_MEAN, places=1)
        low, high = m.mean_ci(0.05)
        self.assertLess(low, TRUE_MEAN)
        self.assertGreater(high, TRUE_MEAN)

    def test_ci_mean_uniform(self):
        X = np.random.uniform(-1, 1, 100)
        TRUE_MEAN = 0
        EMPIRICAL_MEAN = np.mean(X)
        m = CDFModel()
        m.fit(X)
        self.assertAlmostEqual(m.mean(), EMPIRICAL_MEAN, places=1)
        low, high = m.mean_ci(0.05)
        self.assertLess(low, TRUE_MEAN)
        self.assertGreater(high, TRUE_MEAN)