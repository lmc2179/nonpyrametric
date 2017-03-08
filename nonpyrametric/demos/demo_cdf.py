from nonpyrametric import cdf_model
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pprint import pprint

normal_rvs = np.random.normal(0, 1, 100)
m = cdf_model.CDFModel()
m.fit(normal_rvs)
X = np.linspace(-2, 2)
true_cdf = [norm.cdf(x, 0, 1) for x in X]
low, high = zip(*tqdm([m.cdf_ci(x, alpha=0.01) for x in X]))
approx_cdf = [m.cdf(x) for x in X]
pprint(list(zip(low, true_cdf, high)))
plt.plot(X, true_cdf)
plt.plot(X, low)
plt.plot(X, high)
plt.plot(X, approx_cdf)
plt.show()