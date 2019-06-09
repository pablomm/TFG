

import skfda
import matplotlib.pyplot as plt
import numpy as np






fd = skfda.datasets.make_random_warping(shape_parameter=15, n_samples=6, random_state=3)

plt.figure("random-warpings")

fd.plot()

plt.scatter([0, 1], [0, 1], color='maroon')

t = np.linspace(0, 1)
plt.plot(t, t, color='maroon', linestyle='--', label=r'identity $\gamma_{id}$')
plt.legend()

plt.tight_layout()
plt.show()
