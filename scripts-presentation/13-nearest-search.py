# Author: Pablo Marcos Manchón
# License: MIT

# sphinx_gallery_thumbnail_number = 2


import skfda
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from skfda.ml.classification import RadiusNeighborsClassifier
from skfda.misc.metrics import pairwise_distance, lp_distance
plt.style.use('seaborn')

fd1 = skfda.datasets.make_sinusoidal_process(error_std=.0, phase_std=.35,
                                             random_state=0)
fd2 = skfda.datasets.make_sinusoidal_process(phase_mean=1.9, error_std=.0,
                                             random_state=1)

#fd1.plot(color='C0')
#fd2.plot(color='C1')

# Concatenate the two classes in the same FDataGrid
X = fd1.concatenate(fd2)
y = np.array(15*[0] + 15*[1])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    shuffle=True, random_state=0)

plt.figure("radius-search")
plt.xlim(0,1)
plt.ylim(-1.5,1.5)

sample = X_test[0]

X_train.plot(color='C0')


# Creation of pairwise distance
l_inf = pairwise_distance(lp_distance, p=np.inf)
distances = l_inf(sample, X_train)[0] # L_inf distances to 'sample'

X_train[distances <= .3].plot(color='C1')


sample.plot(color='red', linewidth=3)



lower = sample - 0.3
upper = sample + 0.3

plt.fill_between(sample.sample_points[0], lower.data_matrix.flatten(),
                 upper.data_matrix[0].flatten(),  alpha=.25, color='C1')

plt.tight_layout()
plt.figure("k-search")
plt.xlim(0,1)
plt.ylim(-1.5,1.5)

X_train.plot(color='C0')

idx = np.argsort(distances)
idx = idx[:6]


X_train[idx].plot(color="C1")
sample.plot(color='red', linewidth=3)

plt.tight_layout()
plt.show()
