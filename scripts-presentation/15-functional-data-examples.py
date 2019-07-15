

import skfda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

from skfda.datasets import fetch_growth
from skfda.representation.interpolation import SplineInterpolator
from skfda.preprocessing.registration import elastic_registration_warping, elastic_registration




#plt.style.use('seaborn')



#### Berkeley study
data = fetch_growth()


fd = data['data']

fd.dataset_label = None
fd.axes_labels = ['age (years)', 'height (cm)']

plt.figure("berkeley-heights")
fd.plot()

plt.xlim(fd.domain_range[0])
plt.tight_layout()


### Trajectories

def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure("trajectories")
ax = p3.Axes3D(fig)



fd1= skfda.datasets.make_sinusoidal_process(n_samples=2, period=1, error_std=0, random_state=1)
fd2 = skfda.datasets.make_sinusoidal_process(n_samples=2, period=.5, error_std=0, random_state=7)
fd3 = skfda.datasets.make_sinusoidal_process(n_samples=2, period=.5, error_std=0, random_state=12)

fd = fd1.concatenate(fd2, fd3, as_coordinates=True)


# Fifty lines of random 3-D lines
data = fd.data_matrix.squeeze().transpose(0,2,1)


# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# Setting the axes properties
ax.set_xlim3d([-1.5, 1.5])
ax.set_xlabel('X')

ax.set_ylim3d([-1.5, 1.5])
ax.set_ylabel('Y')

ax.set_zlim3d([-1.5, 1.5])
ax.set_zlabel('Z')

#ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, 100, fargs=(data, lines),
                                   interval=50, blit=False)
line_ani.save("trajectories.gif", writer='imagemagick')
plt.show()
