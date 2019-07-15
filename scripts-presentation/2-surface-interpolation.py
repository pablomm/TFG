
import skfda
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from skfda.datasets import make_sinusoidal_process
from skfda.representation.interpolation import SplineInterpolator




X, Y, Z = axes3d.get_test_data(1.2)
data_matrix = [Z.T]
sample_points = [X[0,:], Y[:, 0]]


fd = skfda.FDataGrid(data_matrix, sample_points)

plt.figure("surface-bilinear")
fig, ax = fd.plot(alpha=.9)
fd.scatter(ax=ax, color="maroon")

plt.tight_layout()

plt.figure("surface-bicubic")
fd.interpolator = SplineInterpolator(interpolation_order=3)

fig, ax = fd.plot(alpha=.9)
fd.scatter(ax=ax, color="maroon")
plt.tight_layout()

plt.figure("surface-bicubic-dx")
fig, ax = fd.plot(derivative=(1,0), alpha=.9)
plt.tight_layout()

plt.show()
