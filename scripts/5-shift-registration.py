
import skfda
import matplotlib.pyplot as plt


fd = skfda.datasets.make_sinusoidal_process(random_state=1)

#fd.plot()

basis = skfda.representation.basis.Fourier(nbasis=11)
fd_basis = fd.to_basis(basis)

plt.figure("sine-waves")
fd_basis.plot()
plt.xlim(fd_basis.domain_range[0])
plt.tight_layout()


fd_registered = skfda.preprocessing.registration.shift_registration(fd_basis)


plt.figure("sine-waves-shifted")
fd_registered.plot()
plt.xlim(fd_registered.domain_range[0])
plt.tight_layout()

plt.show()
