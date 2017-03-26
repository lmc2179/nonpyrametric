import pandas as pd
from nonpyrametric import cdf_model
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

FILENAME = 'Sunspot_reports_from_1749.csv'
TIME = 'time'
SUNSPOTS_MONTH = 'sunspot.month'

df = pd.read_csv(FILENAME)
df = df[df[TIME] >= 1900]
m = cdf_model.CDFModel()
m.fit(df[SUNSPOTS_MONTH])
print('Empirical mean:', np.mean(df[SUNSPOTS_MONTH]))
print('Estimated mean:', m.mean())
print('Estimated mean CI-95:', m.mean_ci(0.05))
print('Estimated mean CI-99:', m.mean_ci(0.01))
print('Likelihood of < 30 occurrences (CI-95):', m.cdf_ci(30, 0.05))
print('Quartiles:', )
X = np.linspace(0, max(df[SUNSPOTS_MONTH]))
low, high = zip(*tqdm([m.cdf_ci(x, alpha=0.01) for x in X]))
approx_cdf = [m.cdf(x) for x in X]
plt.plot(X, low)
plt.plot(X, high)
plt.plot(X, approx_cdf)
plt.show()
