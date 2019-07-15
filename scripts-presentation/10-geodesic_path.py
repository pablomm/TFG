


import skfda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from skfda.preprocessing.registration import from_srsf, to_srsf
from matplotlib import animation
plt.style.use('seaborn')

fd = skfda.datasets.make_multimodal_samples(n_samples=1,start=-2, stop=2, random_state=1, modes_location=[-1])
fd2 = 0.6*skfda.datasets.make_multimodal_samples(n_samples=1,start=-2, stop=2, n_modes=2, random_state=1, modes_location=[0.6, 1.1])




q = to_srsf(fd)
q2 = to_srsf(fd2)

cmap = clr.LinearSegmentedColormap.from_list('custom cmap', ['C0','C1'])

v = np.linspace(0, 1, 100)
color = cmap(v)



fig = plt.figure("srsf-geodesic")

ax = plt.axes()

plt.tight_layout()

q.plot(label="$q_1$", linewidth=2)
q2.plot(label="$q_2$", linewidth=2)

line , l2 = ax.get_lines()

def init():
    line.set_data([], [])
    l2.set_data([], [])

    return line,

# animation function.  This is called sequentially
def animate(i):
    if i < 100:
        p = i / 100
        pp = 1 - p
    else:
        pp = (i-100) / 100
        p = 1 - pp
    #print(a,b,c, d)
    ff = p * q + pp * q2

    line.set_data(ff.sample_points[0], ff.data_matrix.squeeze())
    if i < 100:
        j = i
    else:
        j = 199 - i
    line.set_color(color[j])
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)

anim.save("geodesic-q.gif", writer='imagemagick')

plt.show()



fig = plt.figure("srsf-geodesic")

ax = plt.axes()

plt.tight_layout()

fd.plot(label="$f_1$", linewidth=2)
fd2.plot(label="$f_2$", linewidth=2)

line , l2 = ax.get_lines()


# animation function.  This is called sequentially
def animate(i):
    if i < 100:
        p = i / 100
        pp = 1 - p
    else:
        pp = (i-100) / 100
        p = 1 - pp
    #print(a,b,c, d)
    ff = from_srsf(p * q + pp * q2)

    line.set_data(ff.sample_points[0], ff.data_matrix.squeeze())
    if i < 100:
        j = i
    else:
        j = 199 - i
    line.set_color(color[j])
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)

anim.save("geodesic-f.gif", writer='imagemagick')

plt.show()
