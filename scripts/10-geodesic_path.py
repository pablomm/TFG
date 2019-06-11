


import skfda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from skfda.preprocessing.registration import from_srsf, to_srsf


fd = skfda.datasets.make_multimodal_samples(n_samples=1,start=-2, stop=2, random_state=1, modes_location=[-1])
fd2 = 0.6*skfda.datasets.make_multimodal_samples(n_samples=1,start=-2, stop=2, n_modes=2, random_state=1, modes_location=[0.6, 1.1])




q = to_srsf(fd)
q2 = to_srsf(fd2)

cmap = clr.LinearSegmentedColormap.from_list('custom cmap', ['C0','C1'])

v = [.25, .5, .75]
color = cmap(v)

plt.figure("srsf-geodesic")
q.plot(label="$q_1$", linewidth=2)
q2.plot(label="$q_2$", linewidth=2)

for m, c in zip(v, color):
    y = (1 - m)*q + m*q2
    y.plot(color=c)
plt.legend()
plt.tight_layout()

plt.figure("geodesic")
fd.plot(label="$f_1$", linewidth=2)
fd2.plot(label="$f_2$", linewidth=2)

for m, c in zip(v, color):
    y = (1 - m)*q + m*q2
    from_srsf(y).plot(color=c)
plt.legend()
plt.tight_layout()
plt.show()
