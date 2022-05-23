import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

plt.ion()
plt.show()

## Génération des filtres

def g(x):
    if x < 1:
        return x**2
    elif x <= 2:
        return -5 + 11*x - 6*x**2 + x**3
    else:
        return 4*x**-2

gamma = 1.384

def h(x, lmax):
    return gamma * np.exp(-( 10*x/(0.3*lmax) )**4)

def frequencies(lmax, r):
    return np.geomspace(lmax/40, lmax/2, r)

## Obtention d'un coefficient sur la base d'ondelettes

def coef(G, s, node, freq_index, lmax=None):
    if lmax is None:
        G.estimate_lmax()
        lmax = G.lmax
    d = np.zeros_like(s)
    d[node] = 1
    if freq_index == 0:
        return pygsp.filters.Filter(G, lambda x: h(x))



##

l = np.linspace(0, 10, 501)
lmax = 10
r = 7

plt.close('all')
fig, ax = plt.subplots()
ax.plot(l, np.array([h(x, lmax) for x in l]), label="passe-bas")
for w0 in frequencies(lmax, r):
    ax.plot(l, np.array([g(x/w0) for x in l]), label="{:.2f}".format(w0))
ax.legend()

##

G = graphs.DavidSensorNet()
G.estimate_lmax()
B = filters.MexicanHat(G)
vs = np.transpose(B.localize(G.N // 2))
plt.close('all')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.tight_layout()
B.plot(ax=axes[0])
G.plot_signal(vs[5], ax=axes[1], vertex_size=30)








