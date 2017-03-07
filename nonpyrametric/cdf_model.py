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
        eps = sqrt((1/(2*self.n))*log(2/alpha))
        y = self.cdf(x)
        return y - eps, y + eps

    def plot(self, alpha=0.05):
        pass

    def quantile(self, X):
        pass

    def inv_cdf(self, X):
        pass