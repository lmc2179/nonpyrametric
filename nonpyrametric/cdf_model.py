from nonpyrametric.cadlag import CadlagStepFunction
from collections import Counter
import numpy as np
from math import sqrt, log

class CDFModel(object):
    def __init__(self):
        self.empirical_cdf_function = None
        self.n = None

    def fit(self, X):
        data_counter = Counter(X)
        X_cdf = sorted(data_counter.keys())
        Y_cdf = np.cumsum([1.0*data_counter[x]/len(X) for x in X_cdf])
        self.empirical_cdf_function = CadlagStepFunction(X_cdf, Y_cdf)
        self.n = len(X)

    def cdf(self, x):
        return self.empirical_cdf_function(x)

    def cdf_ci(self, x, alpha=0.05):
        eps = sqrt((1./(2*self.n))*log(2/alpha))
        y = self.cdf(x)
        return max(y - eps, 0), min(y + eps, 1)

    def plot(self, alpha=0.05):
        pass

    def quantile(self, q):
        return self.empirical_cdf_function.inverse(q)

    def inv_cdf(self, X):
        pass

    def mean(self):
        X = self.empirical_cdf_function.get_X()
        return sum([(int(X[i] >= 0) - self.cdf(X[i]))*(X[i+1]-X[i]) for i in range(len(X)-1)])

    def mean_ci(self, alpha=0.05):
        X = self.empirical_cdf_function.get_X()
        bases = [(X[i+1]-X[i]) for i in range(len(X)-1)]
        left_sides = [X[i] for i in range(len(X)-1)]
        low, high = zip(*[self.cdf_ci(X[i], alpha) for i in range(len(X)-1)])
        return sum([(int(left >= 0) - h)*base for h, base, left in zip(high, bases, left_sides)]), \
               sum([(int(left >= 0) - l)*base for l, base, left in zip(low, bases, left_sides)])