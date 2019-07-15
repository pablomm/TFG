
import skfda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler


from skfda.preprocessing.registration import elastic_mean, warping_mean, elastic_registration_warping
from skfda.misc.metrics import (pairwise_distance, amplitude_distance,
                                warping_distance)



fd = skfda.datasets.make_multimodal_samples(n_samples=10, random_state=73)
rands = np.random.RandomState(71).normal(1.5, 0.8, len(fd))


fd = fd.copy(data_matrix= (fd.data_matrix[..., 0].T * rands ).T)

plt.figure("unimodal-dataset")
fd.plot()

ylims = plt.ylim()

plt.tight_layout()

plt.figure("means")

fd.mean().plot(label=r"cross-sectional mean")

mean_el = elastic_mean(fd)
mean_el.plot(label=f"elastic mean")
#plt.ylim(ylims)
plt.legend()

plt.tight_layout()
plt.figure("dataset-by-amplitude")




# Get colors
cmap = plt.cm.Reds_r

amp = pairwise_distance(amplitude_distance)


distances = amp(mean_el, fd)[0]
#print(distances)

index = np.argsort(distances)[::-1]
maximun = np.max(distances)
minimun = np.min(distances)

color = cmap((distances - minimun) / (maximun - minimun))
#print(color)


for i in range(len(fd)):
    indice = int(index[i])
    fd[indice].plot(color=color[indice])

plt.tight_layout()
plt.figure("dataset-by-phase")

warpings = elastic_registration_warping(fd, mean_el)
mean_warp = warping_mean(warpings)


pha = pairwise_distance(warping_distance)

distances = pha(mean_warp, warpings)[0]
#print(distances)

maximun = np.max(distances)
minimun = np.min(distances)
color = cmap((distances - minimun) / (maximun - minimun))
index = np.argsort(distances)[::-1]



for i in range(len(fd)):
    indice = int(index[i])
    fd[indice].plot(color=color[indice])

plt.tight_layout()
plt.show()

"""
# Diagram
factor = 1.3
width=1

land = skfda.datasets.make_multimodal_landmarks(n_samples=10, random_state=73).flatten()

angles = np.pi * (.5 + factor *land)

# Vector mean
m_amplitude = np.mean(rands)
m_angle = np.mean(angles)
x_elas = round(m_amplitude*np.cos(m_angle),3)
y_elas = round(m_amplitude*np.sin(m_angle),3)


amplitude_distance = np.abs(rands - m_amplitude)
phase_distance = np.abs(angles - m_angle)
j=0
for distances in [amplitude_distance,phase_distance]:
    minimun = np.min(distances)
    maximun = np.max(distances)
    minmax = (distances - minimun) / (maximun-minimun)
    j+= 1
    colors = cmap(minmax)
    #print("distancias",colors)
    #print("standart", minmax)

    print("%% Start Figure:")
    print(r"\begin{tikzpicture}")

    print(r"%% Axis")
    print(r"\draw[<->] (-2.5,0)--(2.5,0) node[above]{$\operatorname{Re}\{z\}$};")
    print(r"\draw[<->] (0,-1.5)--(0,2.5) node[above]{$\operatorname{Im}\{z\}$};")

    print(r"%%Vectors colored")

    for i in range(len(land)):
        l = angles[i]
        a = rands[i]
        c = 255*colors[i]
        x = round(a*np.cos(angles[i]),3)
        y = round(a*np.sin(angles[i]),3)
        r,g,b = (round(c[0]), round(c[1]), round(c[2]))
        print(r"\definecolor{color%d%d}{RGB}{%d,%d,%d}" % (j, i,r,g,b))
        print(f"\\draw[color{j}{i}, line width={width}pt,-stealth](0,0)--({x},{y});")

    print(r"%%Karcher mean of vectors")
    print(f"\\draw[line width={width+.2}pt, dashed, -stealth](0,0)--({x_elas},{y_elas});")
    print(r"\end{tikzpicture}")


### Figura Rotacion




print("%% Figure with rotation")
print(r"\begin{tikzpicture}")

print(r"%% Unit circle")
print(r"\draw[thick, dashed] (0, 0) circle (1);")
print(r"\node[text width=0.2] at (0.85,-0.85) {$\mathbb{T}$};")

print(r"%%Vectors colored")

for i in range(len(land)):
    l = angles[i]
    x = round(2*np.cos(l),3)
    y = round(2*np.sin(l),3)

    print(f"\\draw[line width={width}pt,-stealth](0,0)--({x},{y});")

print(r"%%Karcher mean of vectors")
print(f"\\draw[line width={width+.2}pt, dashed, -stealth](0,0)--({round(2*np.cos(0), 3)},{round(2*np.sin(0), 3)});")
print(r"\end{tikzpicture}")
"""
