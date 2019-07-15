

import skfda
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation



plt.style.use('seaborn')


fd = skfda.datasets.make_random_warping(shape_parameter=15, n_samples=6, random_state=3)

plt.figure("random-warpings")

fd.plot()

plt.xlim((-.05, 1.05))
plt.ylim((-.05, 1.05))

plt.scatter([0, 1], [0, 1], color='maroon')

t = np.linspace(0, 1, 100)
plt.plot(t, t, color='maroon', linestyle='--', label=r'identity $\gamma_{id}$')
plt.legend()

plt.tight_layout()

fig = plt.figure("random-warpings-animation")
ax = plt.axes(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))
ax.plot(t, t, color='maroon', linestyle='--', label=r'identity $\gamma_{id}$')
plt.scatter([0, 1], [0, 1], color='maroon')


data = fd[:3].data_matrix.squeeze()

data = np.concatenate((t.reshape((1,100)), data))


line, = ax.plot(t, t, color='C0', lw=2, label=r'$\gamma_i$')
plt.legend()

plt.tight_layout()

def weights(i):

    if i < 30:
        return 1, 0, 0, 0
    elif i <= 80:
        p = (80 - i)/50
        q = 1 - p
        return p, q, 0, 0
    elif i < 130:
        p = (130 - i) / 50
        q = 1 - p
        return 0, p, q, 0

    elif i < 230:
        p = (230 - i) / 100
        q = 1 - p
        return 0, 0, p, q

    elif i < 280:
        p = (280 - i) / 50
        q = 1 - p
        return q, 0, 0, p

def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    a, b, c, d = weights(i)
    #print(a,b,c, d)

    y = a * data[0] + b * data[1] + c * data[2] + d*data[3]
    line.set_data(t, y)
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=280, interval=35, blit=True)

anim.save("warpings.gif", writer='imagemagick')

fig = plt.figure("composition-animation")


ax = plt.axes(xlim=(-0.05, 1.05), ylim=(-1.2, 1.2))

fd = skfda.datasets.make_sinusoidal_process(n_samples=1, phase_std=0, amplitude_std=0, error_std=0)
fd.plot(color='maroon', linestyle='--', label=r'$f_i$')
plt.scatter([0, 1], [0, 0], color='maroon')

wa, wb, wc, wd = skfda.FDataGrid(data, sample_points=t)

line, = ax.plot(t, fd.data_matrix.squeeze(), color='C0', lw=2, label=r'$f_i(\gamma_i(t))$')
plt.legend()

plt.tight_layout()


def init2():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate2(i):
    a, b, c, d = weights(i)
    #print(a,b,c, d)

    w = a * wa +  b * wb + c* wc + d* wd
    f = fd.compose(w)
    line.set_data(t, f.data_matrix.squeeze())
    return line,

anim = animation.FuncAnimation(fig, animate2, init_func=init2,
                               frames=280, interval=35, blit=True)

anim.save("composition.gif", writer='imagemagick')






plt.show()
