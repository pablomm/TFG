



import skfda
import numpy as np
import matplotlib.pyplot as plt



fd = skfda.datasets.make_multimodal_samples(n_samples=1, start=0, modes_location=[.3],
                                            mode_std=.008, random_state=1)
fd2 = skfda.datasets.make_multimodal_samples(n_samples=1, start=0,
                                             mode_std=0.001, modes_location=[0.8], random_state=1)
fd2 = .8 * fd2
fd2 = fd2 + fd

plt.figure("pinching-dataset")
fd.plot(color='C1', label="$f_1$")

fd2.plot(linestyle="--", color='C0', label='$f_2$')
plt.legend()
plt.tight_layout()

t = np.linspace(0, 1, 200)
time = np.copy(t)
eps = .04

a = .4
m = 11.9
idx1 = t < a
idx3 = t > a + eps
idx2 = np.logical_not(np.logical_or(idx1, idx3))



t[idx2] = m*(t[idx2] - a) + a
t[idx3] =  np.linspace(t[idx2][-1], 1, idx3.sum())


warp = skfda.FDataGrid([t], time, domain_range=(0,1))


plt.figure("pinching-warping")
warp.plot()
fd_reg = fd2.compose(warp)
plt.tight_layout()

plt.figure("pinching-effect")
fd_reg = fd_reg.to_grid(fd.sample_points[0])


x1 = fd_reg.data_matrix.squeeze()
x2 = fd.data_matrix.squeeze()
t = fd.sample_points[0]

x1[t > .4358] = x2[t > .4358]

fd.plot(label="$f_1$", color="C1")
y = skfda.FDataGrid([x1], t)
y.plot(linestyle="--", label=r"$f_2 \circ \gamma$", color="C0")

plt.legend()
plt.tight_layout()



plt.show()
