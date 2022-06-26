import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

#from test_signal_decomposition import smoothness_and_gft, show_fourier_decomposition, compare_fourier_decomposition, compare_fourier_decomposition_horizontal

from filter_visualisation import show_filter_results

## Filtres

def snr(s, s2):
    return -10 * np.log10( np.mean((s2-s)**2) / np.mean(s**2) )

def low_pass_filter(G, s, s1, wc=None, order=1):
    G.compute_fourier_basis()
    if wc is None:
        wc = 2 * smoothness_and_gft(G, s)[0]
    h = np.vectorize(lambda w: float(w <= wc))
    f = filters.Filter(G, h)
    s2 = f.filter(s1)
    G.compute_fourier_basis()
    snr_vect = [snr(s1, s), snr(s2, s)]
    show_filter_results(G, G.e, [s, s1, s2], [G.gft(s), G.gft(s1), G.gft(s2)], h, snr_vect, suptitle="Filtre idéal")

def high_pass_filter(G, s, s1, wc=None, order=1):
    G.compute_fourier_basis()
    if wc is None:
        wc = 2 * smoothness_and_gft(G, s)[0]
    h = np.vectorize(lambda w: float(w > wc))
    f = filters.Filter(G, h)
    s2 = f.filter(s1)
    snr_vect = [snr(s, s1), snr(s, s2)]
    show_filter_results(G, G.e, [s, s1, s2], [G.gft(s), G.gft(s1), G.gft(s2)], h, snr_vect, suptitle="Filtre idéal")


## Test

plt.close('all')
W = np.loadtxt("data/GraphBretagneHybr.txt")
coords = np.loadtxt("data/GraphCoords.txt")
temp = np.loadtxt("data/Temperature.txt")
G = graphs.Graph(W)
G.set_coordinates(coords)

s = temp[0] - np.average(temp[0])
s1 = s + np.random.normal(0, 0.5, size=G.N)
low_pass_filter(G, s, s1, wc=2, order=1)


