

import skfda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from skfda.preprocessing.registration import from_srsf, to_srsf
from skfda.datasets import make_random_warping

fd = skfda.datasets.make_multimodal_samples(n_samples=1, n_modes=2, random_state=1)

warping = make_random_warping(n_samples=3, start=-1, stop=1, shape_parameter=25, random_state=1)

warping = warping[1:]



plt.figure("orbit-f")

fd.plot(label=r"$f \circ \gamma_i$")

for w in warping:
    fd.compose(w).plot()

plt.legend()
plt.tight_layout()

plt.figure("orbit-q")

to_srsf(fd).plot(label=r"$(q \circ \gamma_i)\sqrt{\dot \gamma_i}$")

for w in warping:
    to_srsf(fd.compose(w)).plot()

plt.legend()
plt.tight_layout()
plt.show()
