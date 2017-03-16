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

    def quantile(self, X):
        pass

    def inv_cdf(self, X):
        pass

    def mean(self):
        X = self.empirical_cdf_function.get_X()
        return sum([1 - self.cdf(X[i])*(X[i+1]-X[i]) for i in range(len(X)-1)])

    def mean_ci(self, alpha=0.05):
        X = self.empirical_cdf_function.get_X()
        bases = [(X[i+1]-X[i]) for i in range(len(X)-1)]
        low, high = zip(*[self.cdf_ci(X[i], alpha) for i in range(len(X)-1)])
        return sum([l*base for l, base in zip(low, bases)]), sum([h*base for h, base in zip(high, bases)])