import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

plt.ion()
plt.show()
plt.rcParams['figure.autolayout'] = True

## Génération des filtres (Hammond)

wavelet_type = "Hammond"

def g0(x):
    #passe-bande
    if x < 1:
        return x**2
    elif x <= 2:
        return -5 + 11*x - 6*x**2 + x**3
    else:
        return 4*x**-2

g = np.vectorize(lambda x: g0(x)/1.384)

def h(x):
    #passe-bas
    return 1.3 * np.exp(-(2/3*x)**4)

## Génération des filtres (Mexican Hat)

wavelet_type = "Mexican Hat"
g = lambda x: x * np.exp(1-x)
h = lambda x: 1.3 * np.exp(-x**4)

## Calculs sur la famille d'ondelettes

def frequencies(lmax, r):
    return np.geomspace(lmax/40, lmax/2, r)

def coef(G, s, node, freq, lmax=None):
    #un coefficient sur la famille ci-après
    if lmax is None:
        G.estimate_lmax()
        lmax = G.lmax
    d = np.zeros_like(s)
    d[node] = 1
    f = lambda x: (h(40*x/lmax) if freq == 0 else g(x/freq))
    return pygsp.filters.Filter(G, f).filter(d) @ s

def basis(G, r):
    #famille d'ondelettes avec r+1 ondelettes par sommet
    #(1 passe bas et r passe-bandes)
    B = np.zeros((r+1, G.N, G.N))
    G.estimate_lmax()
    lmax = G.lmax
    vfreq = np.concatenate(([0], frequencies(G.lmax, r)))
    for i, freq in enumerate(vfreq):
        f = lambda x: (h(40*x/lmax) if freq == 0 else g(x/freq))
        for node in range(G.N):
            d = np.zeros((G.N))
            d[node] = 1
            B[i, node] = pygsp.filters.Filter(G, f).filter(d)
    return B

def coefficients(G, B, s):
    #coefficients sur toute la famille d'ondelettes
    C = np.zeros((B.shape[0], B.shape[1]))
    for i in range(len(C)):
        for node in range(G.N):
            C[i, node] = B[i, node] @ s
    return C

def matching_pursuit(G, B, s, d, suptitle=""):
    #décompose le signal en lui soustrayant à chaque étape l'ondelette prédominante
    signal = s.copy()
    print("La fonction a pour norme {a:2f}".format(a=(signal@signal)**0.5))
    fig = plt.figure(figsize=(10,8))
    gs = plt.GridSpec(2*(d//2 + 1), 2, height_ratios=[2-k%2 for k in range(2*(d//2 + 1))])
    ax = fig.add_subplot(gs[0])
    G.plot_signal(signal, ax=ax, vertex_size=20)
    for k in range(1,d+1):
        C = coefficients(G, B, signal)
        loc = np.argsort(np.abs(C).flatten())
        ax = fig.add_subplot(gs[k + 2*(k//2)])
        i, node = loc[-1]//G.N, loc[-1]%G.N
        G.plot_signal(B[i, node], ax=ax, vertex_size=20)
        title = "Signal" if k==0 else "Ondelette "+str((i, node))
        ax.set_title(title)
        ax.set_axis_off()
        signal = signal - C[i, node]*B[i,node]/(B[i,node]@B[i,node])

        s = (k%2 == 1)
        ax = fig.add_subplot(gs[k + 2*(k//2) + 2*s-1])
        im = ax.imshow(C)
        fig.colorbar(im, ax=ax)
    fig.suptitle(suptitle)
    C = coefficients(G, B, signal)
    s = (d%2 == 0)
    ax = fig.add_subplot(gs[-1 -s])
    im = ax.imshow(C)
    fig.colorbar(im, ax=ax)

    print("Le reste a pour norme {a:2f}".format(a=(signal@signal)**0.5))

def redundancy(B):
    #redondance des éléments de la famille avec les autres,
    #mesurée comme la somme des corrélations au carré
    Br = B.reshape(-1, B.shape[2])
    Corr = np.corrcoef(Br)
    Corr = np.nan_to_num(Corr)
    Corr *= Corr
    R = np.sum(Corr, axis=1) - 1
    return R.reshape(B.shape[0], -1)

## Illustrations

def show_basis(G, B, node):
    r = B.shape[0]//2
    fig, axes = plt.subplots(2, r, figsize=(4*r,6))
    for i in range(B.shape[0]):
        ax = axes[i//r][i%r]
        G.plot_signal(B[i, node], ax=ax, vertex_size=20)
        ax.set_title("")
        ax.set_axis_off()
    fig.suptitle("Ondelettes de " + wavelet_type)

def show_components(G, B, C, s, suptitle=""):
    loc = np.argsort(np.abs(C).flatten())
    fig = plt.figure(figsize=(10,8))
    gs = plt.GridSpec(3, 2, height_ratios=[2, 2, 1])
    for k in range(4):
        ax = fig.add_subplot(gs[k])
        i, node = loc[-k]//G.N, loc[-k]%G.N
        s1 = s if k==0 else B[i, node]
        G.plot_signal(s1, ax=ax, vertex_size=20)
        title = "Signal" if k==0 else "Ondelette "+str((i, node))
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(suptitle)
    gs = plt.GridSpec(3, 1, height_ratios=[2, 2, 1])
    ax = fig.add_subplot(gs[-1])
    im = ax.imshow(C)
    fig.colorbar(im, ax=ax)

## Visualiser l'ensemble de filtres

l = np.linspace(0, 10, 501)
lmax = 10
r = 7

total = np.array([h(40*x/lmax)**2 + np.sum([g(x/w0)**2 for w0 in frequencies(lmax, r)]) for x in l])

plt.close('all')
fig, ax = plt.subplots()
ax.plot(l, np.array([h(40*x/lmax) for x in l]), label="passe-bas")
for w0 in frequencies(lmax, r):
    ax.plot(l, np.array([g(x/w0) for x in l]), label="{:.2f}".format(w0))
ax.plot(l, total, label="total des carrés")
ax.legend()

## Visualiser un extrait de la base

plt.close('all')
G = graphs.DavidSensorNet()
B = basis(G, 5)
show_basis(G, B, G.N//2)

## Exemple 1 : Dirac

G = graphs.DavidSensorNet()
s = np.zeros((G.N))
s[G.N//2] = 1
B = basis(G, 5)
C = coefficients(G, B, s)
plt.close('all')
show_components(G, B, C, s)
matching_pursuit(G, B, s, 3)

## Exemple 2 : Signal lisse

G = graphs.DavidSensorNet()
s = np.array([np.sin(G.coords[i,0]) for i in range(G.N)])
B = basis(G, 5)
C = coefficients(G, B, s)
plt.close('all')
show_components(G, B, C, s)
matching_pursuit(G, B, s, 3)

## Mesure de la redondance

plt.close('all')
G = graphs.DavidSensorNet()
B = basis(G, 5)
for i in range(3):
    for node in range(i%3, G.N, 3):
        B[i, node] = 0
    for node in range((i+1)%3, G.N, 3):
        B[i, node] = 0
R = redundancy(B)
plt.imshow(R)
plt.colorbar()






