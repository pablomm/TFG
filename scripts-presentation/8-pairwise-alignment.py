"""
Pairwise alignment
==================
Shows the usage of the elastic registration to perform a pairwise alignment.
"""

# Author: Pablo Marcos Manchón
# License: MIT

# sphinx_gallery_thumbnail_number = 5


import skfda
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
from matplotlib import animation
plt.style.use('seaborn')

###############################################################################
# Given any two functions :math:`f` and :math:`g`, we define their
# pairwise alignment or  registration to be the problem of finding a warping
# function :math:`\gamma^*` such that a certain energy term
# :math:`E[f, g \circ \gamma]` is minimized.
#
# .. math::
#   \gamma^*= *{argmin}_{\gamma \in \Gamma} E[f \circ \gamma, g]
#
# In the case of elastic registration it is taken as energy function the
# Fisher-Rao distance with a penalisation term, due to the property of
# invariance to reparameterizations of warpings functions.
#
# .. math::
#   E[f \circ \gamma, g] = d_{FR} (f \circ \gamma, g)
#
# Firstly, we will create two unimodal samples, :math:`f` and :math:`g`,
# defined in [0, 1] wich will be used to show the elastic registration.
# Due to the similarity of these curves can be aligned almost perfectly between
# them.
#

# Samples with modes in 1/3 and 2/3
fd = skfda.datasets.make_multimodal_samples(n_samples=2, modes_location=[1/3,2/3],
                                          random_state=1, start=0, mode_std=.01)

plt.figure("pairwise-alignment-dataset")
fd.plot()
plt.legend(['$f$', '$g$'])
plt.tight_layout()
###############################################################################
# In this example :math:`g` will be used as template and :math:`f` will be
# aligned to it. In the following figure it is shown the result of the
# registration process, wich can be computed using :func:`elastic_registration
# <skfda.preprocessing.registration.elastic_registration>`.
#

f, g = fd[0], fd[1]

# Aligns f to g
fd_align = skfda.preprocessing.registration.elastic_registration(f, g)




plt.figure("pairwise-alignment")
fd.plot()
fd_align.plot(color='C0', linestyle='--')


# Legend
plt.legend(['$f$', '$g$', '$f \\circ \\gamma $'])
plt.tight_layout()


###############################################################################
# The non-linear transformation :math:`\gamma` applied to :math:`f` in
# the alignment can be obtained using  :func:`elastic_registration_warping
# <skfda.preprocessing.registration.elastic_registration_warping>`.
#

# Warping to align f to g
warping = skfda.preprocessing.registration.elastic_registration_warping(f, g)
identity = skfda.FDataGrid([warping.sample_points[0]], warping.sample_points[0])


plt.figure("pairwise-alignment-warping")

# Warping used
warping.plot()

# Plot identity
t = np.linspace(0, 1)
plt.plot(t, t, linestyle='--')

# Legend
plt.legend(['$\\gamma$', '$\\gamma_{id}$'])
plt.tight_layout()


###

fig = plt.figure("pairwise-alignment-animation")
ax = plt.axes()

fd.plot(ax)
line, line2 = ax.get_lines()
#plt.legend(['$f_1$', '$f_2$'])
fd[0].plot(color='C0',linestyle='--')
fd[1].plot(color='C1',linestyle='--')
fd0 = fd[0]
plt.tight_layout()


def init():
    line.set_data([], [])

    return line,

# animation function.  This is called sequentially
def animate(i):
    if i > 100:
        p = 1
        q = 0
    else:
        p = i / 100
        q = 1 - p
    #print(a,b,c, d)
    w = p * warping + q * identity
    ff = fd0.compose(w)

    line.set_data(ff.sample_points[0], ff.data_matrix.squeeze())
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=150, interval=20, blit=True)

anim.save("alignment-animation.gif", writer='imagemagick')

##
###############################################################################
# The transformation necessary to align :math:`g` to :math:`f` will be the
# inverse of the original warping function, :math:`\gamma^{-1}`.
# This fact is a consequence of the use of the Fisher-Rao metric as energy
# function.
#

warping_inverse = skfda.preprocessing.registration.invert_warping(warping)



plt.figure("pairwise-alignment-inverse")
fd.plot(label='$f$')
g.compose(warping_inverse).plot(color='C1', linestyle='--')


# Legend
plt.legend(['$f$', '$g$', '$g \\circ \\gamma^{-1} $'])

plt.tight_layout()
###############################################################################
# The amount of deformation used in the registration can be controlled by using
# a variation of the metric with a penalty term
# :math:`\lambda \mathcal{R}(\gamma)` wich will reduce the elasticity of the
# metric.
#
# The following figure shows the original curves and the result to the
# alignment varying :math:`\lambda` from 0 to 0.2.
#

# Values of lambda
lambdas = np.array([0.001, 0.05, 0.15, 0.2])

# Creation of a color gradient
cmap = clr.LinearSegmentedColormap.from_list('custom cmap', ['C1','C0'])
color = cmap(.2 + 3*lambdas)

plt.figure("penalty-elastic")

for lam, c in zip(lambdas, color):
    # Plots result of alignment
    skfda.preprocessing.registration.elastic_registration(f, g, lam=lam).plot(color=c)


f.plot(color='C0', linewidth=2., label='$f$')
g.plot(color='C1', linewidth=2., label='$g$')

# Legend
plt.legend()
plt.tight_layout()

###############################################################################
# This phenomenon of loss of elasticity is clearly observed in
# the warpings used, since as the term of penalty increases, the functions
# are closer to :math:`\gamma_{id}`.
#

plt.figure("penalty-elastic-warping")

for lam, c in zip(lambdas, color):
    skfda.preprocessing.registration.elastic_registration_warping(f, g, lam=lam).plot(color=c)

# Plots identity
plt.plot(t,t,  color='C0', linestyle="--")

plt.tight_layout()
plt.show()


###############################################################################
# * Srivastava, Anuj & Klassen, Eric P. (2016). Functional and shape data
#   analysis. In *Functional Data and Elastic Registration* (pp. 73-122).
#   Springer.
#
# * J. S. Marron, James O. Ramsay, Laura M. Sangalli and Anuj Srivastava (2015).
#   Functional Data Analysis of Amplitude and Phase Variation.
#   Statistical Science 2015, Vol. 30, No. 4
