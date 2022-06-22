import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

from test_signal_decomposition import smoothness_and_gft, show_fourier_decomposition, compare_fourier_decomposition
from prony2 import prony

##

def compare_filters(wc, order, P, Q, ax, sep=1, wmax=None):
    if wmax is None:
        wmax = 5*wc

    ax.plot([0, wc], [1, 1], color='C2')
    ax.plot([wc, wc], [0, 1], color='C2')
    ax.plot([wc, wmax], [0, 0], color='C2', label="Filtre idéal")

    x = np.linspace(0, wmax, 200)
    y = (x <= wc)
    p = np.polyfit(x, y, order)
    ax.plot(x, np.polyval(p, x), color='C0', label="Filtre polynomial (10)")

    vx = np.concatenate((np.linspace(0, wc-sep/2, 10), np.linspace(wc+sep/2, wmax, 10)))
    vy = (vx <= wc) + 1e-6
    p, q = prony(vx, vy, P, Q)
    f = lambda t: np.polyval(q, t) / np.polyval(p, t)
    l = np.linspace(np.min(vx)-0.1, np.max(vx)+0.1, 201)
    ax.plot(l, np.array([f(t) for t in l]), color='C1', label="Filtre rationnel (6, 4)")
    ax.plot(vx, vy, linestyle='None', color='C1', marker='.')
    ax.set_ylim(-0.1, 1.1)
    ax.legend()

##

plt.close('all')
fig, ax = plt.subplots()
compare_filters(4.35, 10, 6, 4, ax, wmax=14, sep=1)