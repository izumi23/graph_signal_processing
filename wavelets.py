import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

plt.ion()
plt.show()

## Génération des filtres

gamma = 1.384

def g0(x):
    if x < 1:
        return x**2
    elif x <= 2:
        return -5 + 11*x - 6*x**2 + x**3
    else:
        return 4*x**-2

g = np.vectorize(lambda x: g0(x)/gamma)

def h(x, lmax):
    return 1.3 * np.exp(-( 8*x/(0.3*lmax) )**4)

def frequencies(lmax, r):
    return np.geomspace(lmax/40, lmax/2, r)

## Calculs sur la base d'ondelettes

def coef(G, s, node, freq, lmax=None):
    #un seul coefficient
    if lmax is None:
        G.estimate_lmax()
        lmax = G.lmax
    d = np.zeros_like(s)
    d[node] = 1
    f = lambda x: (h(x, lmax) if freq == 0 else g(x/freq))
    return pygsp.filters.Filter(G, f).filter(d) @ s

def basis(G, r):
    B = np.zeros((r+1, G.N, G.N))
    G.estimate_lmax()
    vfreq = np.concatenate(([0], frequencies(G.lmax, r)))
    for i, freq in enumerate(vfreq):
        f = lambda x: (h(x, lmax) if freq == 0 else g(x/freq))
        for node in range(G.N):
            d = np.zeros((G.N))
            d[node] = 1
            B[i, node] = pygsp.filters.Filter(G, f).filter(d)
    return B

def coefficients(G, s, r, B=None):
    #coefficients sur toute la famille d'ondelettes
    C = np.zeros((r+1, G.N))
    if B is None:
        B = basis(G, r)
    for i in range(r+1):
        for node in range(G.N):
            C[i, node] = B[i, node] @ s
    return C

## Visualiser l'ensemble de filtres

l = np.linspace(0, 10, 501)
lmax = 10
r = 7

total = np.array([h(x, lmax)**2 + np.sum([g(x/w0)**2 for w0 in frequencies(lmax, r)]) for x in l])

plt.close('all')
fig, ax = plt.subplots()
ax.plot(l, np.array([h(x, lmax) for x in l]), label="passe-bas")
for w0 in frequencies(lmax, r):
    ax.plot(l, np.array([g(x/w0) for x in l]), label="{:.2f}".format(w0))
ax.plot(l, total, label="total des carrés")
ax.legend()

## Exemple 1 : Dirac

G = graphs.DavidSensorNet()
s = np.zeros((G.N))
s[G.N//2] = 1
C = coefficients(G, s, 5)
plt.close('all')
plt.imshow(C)

## Exemple 2 : Signal lisse

G = graphs.DavidSensorNet()
s = np.array([np.sin(G.coords[i,0]) for i in range(G.N)])
C = coefficients(G, s, 5)
plt.close('all')
plt.imshow(C)









