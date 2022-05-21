import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

from test_signal_decomposition import smoothness_and_gft, show_fourier_decomposition, compare_fourier_decomposition

##

def polyfilter(wc, order, wmax=None):
    if wmax is None:
        wmax = 5*wc
    x = np.linspace(0, wmax, 200)
    y = (x <= wc)
    p = np.polyfit(x, y, order)
    return p

def apply_polyfilter(G, s, p):
    H = np.zeros((G.N, G.N))
    for pv in p:
        H = G.L @ H + pv*np.eye(G.N)
    return H @ s

def snr(s2, s):
    return -10 * np.log10( np.mean((s2-s)**2) / np.mean(s**2) )

def low_pass_filter(G, s, s1, wc=None, wmax=None, order=1):
    if wc is None:
        G.estimate_lmax()
        wc = G.lmax/2
    p = polyfilter(wc, order, wmax)
    s2 = apply_polyfilter(G, s1, p)
    suptitle = "SNR = {:.1f} --> {:.1f} dB".format(snr(s1, s), snr(s2, s))
    h = lambda w: np.polyval(p, w)
    compare_fourier_decomposition([G, G, G], [s, s1, s2], suptitle=suptitle, h=h)


## On vérifie que le polynôme est bon

p = polyfilter(10, 20, 21)
fig, ax = plt.subplots()
l = np.linspace(0, 20, 201)
ax.plot(l, np.polyval(p, l))
ax.set_ylim([-0.1, 1.1])

## Exemple 1 : Logo

G = graphs.Logo()
G.compute_fourier_basis()
s = np.sum(G.U[:, :G.N//3], axis=1)
s1 = s + np.random.normal(0, 0.1*np.max(np.abs(s)), size=G.N)
low_pass_filter(G, s, s1, wc=4.4, wmax=14, order=15)

## Exemple 2 : Bretagne

H = np.loadtxt("data/GraphBretagneHybr.txt")
coords = np.loadtxt("data/GraphCoords.txt")
s = np.loadtxt("data/Temperature.txt")

G = graphs.Graph(H)
G.set_coordinates(coords)

k = np.argmin([smoothness_and_gft(G, s[i])[0] for i in range(len(s))])

s1 = s[k] + np.random.normal(0, 3, size=G.N)
low_pass_filter(G, s[k], s1, wc=2, wmax=8.5, order=15)

