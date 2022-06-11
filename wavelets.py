import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

plt.ion()
plt.show()
plt.rcParams['figure.autolayout'] = True

## Génération des filtres (Hammond)

wavelet_type = "de Hammond"

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
    return np.exp(-(2/3*x)**4)

## Génération des filtres (Mexican Hat)

wavelet_type = "Mexican Hat"
g = lambda x: x * np.exp(1-x)
h = lambda x: np.exp(-x**4)

## Calculs sur la famille d'ondelettes

def frequencies(lmax, r):
    return np.geomspace(lmax/40, lmax/2, r)

def impulse_basis(G, r):
    #famille d'ondelettes normées issues de filtres, contenant
    #r+1 ondelettes par sommet (1 passe bas et r passe-bandes)
    B = np.zeros((G.N*(r+1), G.N))
    G.estimate_lmax()
    lmax = G.lmax
    vfreq = np.concatenate(([0], frequencies(G.lmax, r)))
    for l, freq in enumerate(vfreq):
        f = lambda x: (h(40*x/lmax) if freq == 0 else g(x/freq))
        for node in range(G.N):
            d = np.zeros((G.N))
            d[node] = 1
            d = pygsp.filters.Filter(G, f).filter(d)
            B[G.N*l + node] = d/np.sqrt(np.sum(d**2))
    return B

def coefficients(B, s):
    #coefficients sur toute la famille d'ondelettes
    return B @ s

def snr(s, s_compr):
    #signal-to-noise ratio (logarithmic)
    return -10 * np.log10( 1e-14 + np.mean((s_compr-s)**2) / np.mean(s**2) )

def matching_pursuit(G, B, s, nb_coef=None):
    if nb_coef is None:
        nb_coef = G.N
    s_compr = np.zeros_like(s) #compressed
    res = s.copy() #residue
    decomp = np.zeros((nb_coef, 2))
    snr_vect = np.zeros((nb_coef))
    for k in range(nb_coef):
        c = coefficients(B, res)
        n = np.argmax(np.abs(c))
        decomp[k] = [n, c[n]]
        s_compr += c[n] * B[n]
        res -= c[n] * B[n]
        snr_vect[k] = snr(s, s_compr)
    return decomp, snr_vect

def redundancy(B):
    #redondance des éléments de la famille avec les autres,
    #mesurée comme la somme des corrélations au carré
    Corr2 = np.nan_to_num(np.corrcoef(B))**2
    return np.sum(Corr2, axis=1) - 1

## Illustrations

def show_impulse_basis(G, B, node):
    r = B.shape[0]//G.N//2
    fig, axes = plt.subplots(2, r, figsize=(4*r,6))
    for l in range(B.shape[0]//G.N):
        ax = axes[l//r][l%r]
        G.plot_signal(B[l*G.N + node], ax=ax, vertex_size=20)
        ax.set_title("$u_{{{}}}^{{{}}}$".format(node, l))
        ax.set_axis_off()
    fig.suptitle("Ondelettes " + wavelet_type + ", centrées en " + str(node))

def show_components(G, B, s, decomp, snr_vect, nb_coef=5, suptitle=None):
    mp_coef = len(decomp)
    fig = plt.figure(figsize=(4*((nb_coef+2)//2) ,8))
    #affichage du signal et de ses nb_coef composantes les plus
    #importantes
    gs = plt.GridSpec(3, (nb_coef+2)//2, height_ratios=[2, 2, 1])
    ax = fig.add_subplot(gs[0])
    G.plot_signal(s, ax=ax, vertex_size=20)
    ax.set_title("Signal")
    ax.set_axis_off()
    for k in range(nb_coef):
        [n, c] = decomp[k]
        n = int(n)
        ax = fig.add_subplot(gs[k+1])
        G.plot_signal(B[n], ax=ax, vertex_size=20)
        title = "$\\hat s_{{{}}}^{{{}}} = {:.3f}$".format(n%G.N, n//G.N, c)
        ax.set_title(title)
        ax.set_axis_off()
    if suptitle is None:
        suptitle = "Reconstruction du signal (de dimension {}) avec {} vecteurs dans la base ".format(G.N, mp_coef) + wavelet_type
    fig.suptitle(suptitle)

    #affichage de chaque coefficient dans la base
    gs = plt.GridSpec(3, 1, height_ratios=[2, 2, 1])
    gs = gs[-1].subgridspec(1, 2)
    mk = '.' if G.N < 100 else 'None'
    ax = fig.add_subplot(gs[-2])
    ax.plot(decomp[:,0], np.abs(decomp[:,1]), linestyle='None', marker=mk)
    for [n, c] in decomp:
        ax.plot([n, n], [0, np.abs(c)], color='C0')
    ax.set_title("Coefficients dans la décomposition")
    #SNR en prenant uniquement les n composantes les plus importantes
    ax = fig.add_subplot(gs[-1])
    ax.plot(np.arange(1, mp_coef+1), snr_vect, marker=mk)
    ax.set_ylim(-2, 40)
    ax.set_title("SNR vs nombre de composantes utilisées")

## Visualiser l'ensemble de filtres

l = np.linspace(0, 10, 501)
lmax = 10
r = 5

total = np.array([h(40*x/lmax)**2 + np.sum([g(x/w0)**2 for w0 in frequencies(lmax, r)]) for x in l])

plt.close('all')
fig, ax = plt.subplots()
ax.plot(l, np.array([h(40*x/lmax) for x in l]), label="passe-bas")
for w0 in frequencies(lmax, r):
    ax.plot(l, np.array([g(x/w0) for x in l]), label="{:.2f}".format(w0))
ax.plot(l, total, label="total des carrés")
ax.legend()

## Visualiser un extrait de la base impulsionnelle

plt.close('all')
G = graphs.DavidSensorNet()
B = impulse_basis(G, 5)
show_impulse_basis(G, B, G.N//2)

## Exemple 1 : Dirac

G = graphs.DavidSensorNet()
s = np.zeros((G.N))
s[G.N//2] = 1
B = impulse_basis(G, 5)
decomp, snr_vect = matching_pursuit(G, B, s, G.N//2)
plt.close('all')
show_components(G, B, s, decomp, snr_vect)

## Exemple 2 : Signal lisse

G = graphs.Logo()
s = np.array([np.sin(0.01*G.coords[i,0]) for i in range(G.N)])
s = s - np.average(s)
B = impulse_basis(G, 5)
decomp, snr_vect = matching_pursuit(G, B, s, G.N//2)
plt.close('all')
show_components(G, B, s, decomp, snr_vect)

## Exemple 3 : Bretagne

H = np.loadtxt("data/GraphBretagneHybr.txt")
coords = np.loadtxt("data/GraphCoords.txt")
temp = np.loadtxt("data/Temperature.txt")
G = graphs.Graph(H)
G.set_coordinates(coords)

s = temp[0]
B = impulse_basis(G, 5)
decomp, snr_vect = matching_pursuit(G, B, s, G.N)
plt.close('all')
show_components(G, B, s, decomp, snr_vect)

## Exemple 4 : Bruit

H = np.loadtxt("data/GraphBretagneHybr.txt")
coords = np.loadtxt("data/GraphCoords.txt")
G = graphs.Graph(H)
G.set_coordinates(coords)

s = np.random.normal(size=G.N)
B = impulse_basis(G, 5)
decomp, snr_vect = matching_pursuit(G, B, s, G.N)
plt.close('all')
show_components(G, B, s, decomp, snr_vect)

## Mesure de la redondance

plt.close('all')
G = graphs.DavidSensorNet()
B = impulse_basis(G, 5)
for l in range(3):
    for node in range(l%3, G.N, 3):
        B[l*G.N + node] = 0
    for node in range((l+1)%3, G.N, 3):
        B[l*G.N, node] = 0
R = redundancy(B)
plt.imshow(R.reshape(-1, G.N))
plt.colorbar()
