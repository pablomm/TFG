
import skfda
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from skfda.datasets import make_sinusoidal_process
from skfda.representation.interpolation import SplineInterpolator

plt.style.use('seaborn')

fd = make_sinusoidal_process(n_samples=1, n_features=6, random_state=3) + 1.2
fd.interpolator = SplineInterpolator(interpolation_order=3)


plt.figure("original-function")
fd.scatter(label='original values')
fd.plot(label='interpolation', color='C0', linestyle="--")
plt.legend()

x = fd.sample_points[0]
y = fd.data_matrix.squeeze()

ymin, ymax = plt.ylim()

ymin = 0

for i in range(len(x)):
    plt.plot([x[i], x[i]], [-1.2, y[i]], color='C0', linewidth=1)


plt.ylim(ymin, ymax)
plt.yticks([])

plt.tight_layout()
plt.figure("function-resampled")
fdr = fd.to_grid(np.linspace(0, 1, 13))
fdr.scatter(label='values resampled')
fdr.plot(label='interpolation', color='C0', linestyle="--")
plt.legend()

x = fdr.sample_points[0]
y = fdr.data_matrix.squeeze()

for i in range(len(x)):
    plt.plot([x[i], x[i]], [-1.2, y[i]], color='C0', linewidth=1)

plt.ylim(ymin, ymax)

plt.yticks([])
plt.tight_layout()

plt.figure("linear-interpolation")

fd.interpolator = SplineInterpolator(interpolation_order=1)

fd.plot()
fd.scatter()

plt.tight_layout()



plt.figure("spline-interpolation")
labels = ["linear", "cuadratic", "cubic"]

for i in range(1, 4):
    fd.interpolator = SplineInterpolator(interpolation_order=i)

    fd.plot(label=labels[i-1])

fd.scatter()
plt.legend()
plt.tight_layout()

plt.figure("spline-derivatives-interpolation")

for i in range(1, 4):
    fd.interpolator = SplineInterpolator(interpolation_order=i)

    fd.plot(derivative=1, label=labels[i-1])

#fd.scatter()
plt.legend()
plt.tight_layout()



# Sample with noise
fd_smooth = skfda.datasets.make_sinusoidal_process(n_samples=1, n_features=30,
                                                 random_state=1, error_std=.3)

# Cubic interpolator
plt.figure("smoothing-splines")
fd_smooth.interpolator = SplineInterpolator(interpolation_order=3)

fd_smooth.plot(label="cubic spline")

# Smooth interpolation
fd_smooth.interpolator = SplineInterpolator(interpolation_order=3,
                                            smoothness_parameter=1.5)

fd_smooth.plot(label="smoothing cubic spline", linewidth=2)


fd_smooth.scatter(color='maroon')
plt.yticks([])
plt.legend()

plt.tight_layout()

#######
# Cubic interpolator
plt.figure("smoothing-splines-values")

for a in [0.5, 1.5, 2.5]:
    # Smooth interpolation
    fd_smooth.interpolator = SplineInterpolator(interpolation_order=3,
                                                smoothness_parameter=a)

    fd_smooth.plot(label=f"$\\alpha={a}$", linewidth=2)


fd_smooth.scatter(color='maroon')
plt.yticks([])
plt.legend()

plt.tight_layout()

plt.show()
