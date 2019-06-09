

import skfda
import matplotlib.pyplot as plt
import numpy as np




fd = skfda.datasets.make_multimodal_samples(n_samples=3, n_modes=2, std=.008,
                                          mode_std=.008, random_state=30)

fd = np.sqrt(fd)



landmarks = skfda.datasets.make_multimodal_landmarks(n_samples=3, n_modes=2, std=.008,
                                           random_state=30).squeeze()

plt.figure("landmark-dataset")
fd.plot()

for l in landmarks:
    plt.scatter(l, [1, 1])


plt.tight_layout()

warping = skfda.preprocessing.registration.landmark_registration_warping(fd, landmarks)

plt.figure("landmark-warping")

# Plots warping
warping.plot()

lm = landmarks.mean(axis=0)

# Plot landmarks
for i in range(fd.nsamples):
    plt.scatter(lm, landmarks[i], color=f"C{i}")
    plt.scatter([-1, 1], [-1, 1], color=f"C{i}")

    plt.plot(2*[lm[0]], [-1, landmarks[i, 0]], color=f"C{i}", linestyle='--', linewidth=1)
    plt.plot(2*[lm[1]], [-1, landmarks[i, 1]], color=f"C{i}", linestyle='--', linewidth=1)
    plt.plot([-1, lm[0]], 2*[landmarks[i, 0]], color=f"C{i}", linestyle='--', linewidth=1)
    plt.plot([-1, lm[1]], 2*[landmarks[i, 1]], color=f"C{i}", linestyle='--', linewidth=1)

plt.ylim(-1, 1)
plt.xlim(-1, 1)
plt.tight_layout()



plt.figure("landmark-registration")

fd_registered = skfda.preprocessing.registration.landmark_registration(fd, landmarks)
fd_registered.plot()

plt.scatter(lm, [1, 1], color='maroon', label="common landmarks")

plt.tight_layout()

plt.show()
