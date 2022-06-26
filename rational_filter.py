import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

#from test_signal_decomposition import smoothness_and_gft, show_fourier_decomposition, compare_fourier_decomposition

from prony2 import prony
from filter_visualisation import show_filter_results

##

def ratfilter(wc, P, Q, sep=1, wmax=None, plot=False):
    if wmax is None:
        wmax = 5*wc
    vx = np.concatenate((np.linspace(0, wc-sep/2, 10), np.linspace(wc+sep/2, wmax, 10)))
    vy = (vx <= wc) + 0.01
    return prony(vx, vy, P, Q, plot)

def polyval(p, L):
    N = L.shape[0]
    H = np.zeros((N, N))
    for pv in p:
        H = L @ H + pv*np.eye(N)
    return H

def apply_ratfilter(p, q, L, s):
    return np.linalg.solve(polyval(p, L), polyval(q, L) @ s)

def snr(s2, s):
    return -10 * np.log10( np.mean((s2-s)**2) / np.mean(s**2) )

def low_pass_filter(G, s, s1, wc=None, wmax=None, P=6, Q=4, sep=1):
    if wc is None:
        G.estimate_lmax()
        wc = G.lmax/2
    p, q = ratfilter(wc, P, Q, sep=sep, wmax=wmax)
    s2 = apply_ratfilter(p, q, G.L, s1)
    snr_vect = [snr(s1, s), snr(s2, s)]
    h = lambda w: np.polyval(q, w) / np.polyval(p, w)
    G.compute_fourier_basis()
    show_filter_results(G, G.e, [s, s1, s2], [G.gft(s), G.gft(s1), G.gft(s2)], h, snr_vect, suptitle="Filtre rationnel")


## On vérifie que le polynôme est bon

p, q = ratfilter(10, 6, 4, sep=2, wmax=21, plot=True)
p, q = ratfilter(2, 1, 0, wmax=40, plot=True)

## Exemple 1 : Logo

G = graphs.Logo()
G.compute_fourier_basis()
s = np.sum(G.U[:, :G.N//3], axis=1)
s1 = s + np.random.normal(0, 0.1*np.max(np.abs(s)), size=G.N)
low_pass_filter(G, s, s1, wc=4.35, wmax=14, sep=0.5)

## Exemple 2 : Bretagne

H = np.loadtxt("data/GraphBretagneHybr.txt")
coords = np.loadtxt("data/GraphCoords.txt")
s = np.loadtxt("data/Temperature.txt")

G = graphs.Graph(H)
G.set_coordinates(coords)

#k = np.argmin([smoothness_and_gft(G, s[i])[0] for i in range(len(s))])
k = 0

s1 = s[k] + np.random.normal(0, 3, size=G.N)
low_pass_filter(G, s[k], s1, wc=2, wmax=8.5)

