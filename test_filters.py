import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

from test_signal_decomposition import show_fourier_decomposition, compare_fourier_decomposition

## Filtre

def filter(G, s, wc=None, order=1):
    G.compute_fourier_basis()
    if wc is None:
        s_hat = G.gft(s)
        s_hat[0] = 0
        s0 = G.igft(s_hat)
        smoothness = (s0 @ G.L @ s0) / (s0 @ s0)
        wc = 2 * smoothness
    h = lambda w: 1 / np.sqrt(1 + (w/wc)**(2*n))
    f = filters.Filter(G, h)
    s1 = s + np.random.normal(0, 0.25, size=G.N)
    s2 = f.filter(s)
    compare_fourier_decomposition([G, G, G], [s, s1, s2], h)

## Test sur le logo GSP

plt.close('all')
G = graphs.Logo()
s = np.cos(0.01*np.arange(G.N))
filter(G, s, 4)

## Test sur la Bretagne (nul)

W = np.loadtxt("data/GraphBretagne.txt")
coords = np.loadtxt("data/GraphCoords.txt")
s = np.loadtxt("data/Temperature.txt")

G = graphs.Graph(W)
G.set_coordinates(coords)

filter(G, s[0])
