
import skfda
import numpy as np
import matplotlib.pyplot as plt
from skfda.datasets import fetch_growth
from skfda.representation.interpolation import SplineInterpolator
from skfda.preprocessing.registration import elastic_registration_warping, elastic_registration


data = fetch_growth()


fd = data['data']

fd = fd[data['target'] == 0]

fd = fd.derivative()
fd.interpolator = SplineInterpolator(3)
fd.dataset_label = None
fd.axes_labels = ["age", r"$\partial \, height \, / \, \partial \, age$"]

plt.figure("berkeley-males")
fd.plot()

plt.xlim(fd.domain_range[0])

plt.tight_layout()

plt.figure("berkeley-warping")
w = elastic_registration_warping(fd)
w.axes_labels = ['age', 'age']
w.plot()
plt.tight_layout()
plt.figure("berkeley-registered")
fd.compose(w).plot()

plt.tight_layout()
plt.show()
