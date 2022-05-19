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

def apply_polyfilter(G, s0, p):
    s = s0.copy()
    for pv in p:
        s = G.L @ s + pv*s
    return s

def snr(s2, s):
    return -10 * np.log10( np.mean((s2-s)**2) / np.mean(s**2) )

def low_pass_filter(G, s, s1, wc=None, wmax=None, order=1):
    if wc is None:
        G.estimate_lmax()
        wc = G.lmax/2
    p = polyfilter(wc, order, wmax)
    s2 = apply_polyfilter(G, s1, p)
    suptitle = "SNR = {:.1f} --> {:.1f} dB".format(snr(s1, s), snr(s2, s))
    compare_fourier_decomposition([G, G, G], [s, s1, s2], suptitle=suptitle)

##

p = polyfilter(10, 20, 21)
fig, ax = plt.subplots()
ax.plot(l, np.polyval(p, l))
ax.set_ylim([-0.1, 1.1])

##

G = graphs.Logo()
s = np.array([G.coords[i][0] for i in range(G.N)])
s1 = s + np.random.normal(0, 3, size=G.N)
#low_pass_filter(G, s, s1)
p = polyfilter(10, 10, 20)
s2 = apply_polyfilter(G, s1, p)



