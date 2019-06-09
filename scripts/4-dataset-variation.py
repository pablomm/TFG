

import numpy as np
import matplotlib.pyplot as plt

def sine(t, pha=None, amp=None):
    """ sin(t) function with phase and amplitude variation """

    n = len(pha) if pha is not None else len(amp)

    if pha is not None and amp is not None:
        pha = np.asarray(pha).T
        amp = np.diag(amp)
    elif amp is None:
        pha = np.asarray(pha).T
        amp = np.diag(np.ones(n))
    elif pha is None:
        amp = np.diag(amp)
        pha = np.zeros(n).T

    t = np.outer(t, np.ones(n))

    return np.sin(pha + t) @ amp


if __name__ == "__main__":

    pha = np.arange(-0.9, 0.9, 0.4)  # Phase variation
    amp = np.arange(0.4, 1.1, 0.15)  # Amplitud variation
    lims = (0, 2*np.pi)  # Limits of the interval
    n = 100  #  Number of points

    t = np.linspace(lims[0], lims[1], n)
    data_phase = sine(t, pha=pha)
    data_amp = sine(t, amp=amp)

    # Makes a plot with the data and the means

    plt.figure("phase-variation")
    plt.plot(t, data_phase, color='teal')
    plt.plot(t, data_phase.mean(axis=1), color='maroon', linestyle='dashed',
             label='mean')
    plt.xlim(lims)
    plt.yticks([])
    plt.xticks([])
    plt.legend()
    plt.tight_layout()

    plt.figure("amplitude-variation")
    plt.plot(t, data_amp, color='teal')
    plt.plot(t, data_amp.mean(axis=1), color='maroon', linestyle='dashed',
             label='mean')
    plt.xlim(lims)
    plt.yticks([])
    plt.xticks([])
    plt.legend()
    plt.tight_layout()

    plt.show()
