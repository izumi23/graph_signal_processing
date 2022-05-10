import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

from test_signal_decomposition import smoothness_and_gft, show_fourier_decomposition, compare_fourier_decomposition, compare_fourier_decomposition_horizontal

## Filtres

def snr(s2, s):
    return -10 * np.log10( np.mean((s2-s)**2) / np.mean(s**2) )

def low_pass_filter(G, s, s1, wc=None, order=1):
    G.compute_fourier_basis()
    if wc is None:
        wc = 2 * smoothness_and_gft(G, s)[0]
    h = lambda w: 1 / np.sqrt(1 + (w/wc)**(2*order))
    f = filters.Filter(G, h)
    s2 = f.filter(s1)
    suptitle = "SNR = {:.1f} --> {:.1f} dB".format(snr(s1, s), snr(s2, s))
    compare_fourier_decomposition_horizontal([G, G, G], [s, s1, s2], h, suptitle=suptitle)

def high_pass_filter(G, s, s1, wc=None, order=1):
    G.compute_fourier_basis()
    if wc is None:
        wc = 2 * smoothness_and_gft(G, s)[0]
    h = lambda w: 1 / np.sqrt(1 + (wc/(w+1e-7))**(2*order))
    f = filters.Filter(G, h)
    s2 = f.filter(s1)
    suptitle = "Anomalie en {}".format(np.argmax(np.abs(s2)))
    compare_fourier_decomposition([G, G, G], [s, s1, s2], h, suptitle=suptitle)




## Construction des graphes

W = np.loadtxt("data/GraphBretagne.txt")
coords = np.loadtxt("data/GraphCoords.txt")
s = np.loadtxt("data/Temperature.txt")

G = graphs.Graph(W)
G.set_coordinates(coords)

C = np.corrcoef(s, rowvar=False) - np.eye(len(W))
t = 0.92
Wc = C * (C >= t)
G1 = graphs.Graph(Wc)
G1.set_coordinates(coords)

C = np.corrcoef(s, rowvar=False) - np.eye(len(W))
for i in range(len(W)):
    for j in range(len(W)):
        if abs(W[i,j]) > 1e-14:
            W[i,j] = C[i,j]
G2 = graphs.Graph(W)
G2.set_coordinates(coords)


## Test 0
k = np.argmin([smoothness_and_gft(G, s[i])[0] for i in range(len(s))])

s1 = s[k] + np.random.normal(0, 3, size=G.N)
low_pass_filter(G, s[k], s1, wc=2, order=1)

## Test 1

plt.close('all')
C = np.corrcoef(s, rowvar=False) - np.eye(len(W))
t = 0.92
Wc = C * (C >= t)
G1 = graphs.Graph(Wc)
G1.set_coordinates(coords)

k = np.argmin([smoothness_and_gft(G1, s[i])[0] for i in range(len(s))])

s1 = s[k] + np.random.normal(0, 3, size=G.N)
low_pass_filter(G1, s[k], s1, wc=2, order=1)

## Test 2

W = np.loadtxt("data/GraphBretagne.txt")
coords = np.loadtxt("data/GraphCoords.txt")
s = np.loadtxt("data/Temperature.txt")

plt.close('all')


k = np.argmin([smoothness_and_gft(G2, s[i])[0] for i in range(len(s))])

s1 = s[k] + np.random.normal(0, 3, size=G.N)
low_pass_filter(G, s[k], s1, wc=2, order=1)

# s1 = s[0] + np.random.normal(0, 1, size=G.N)
# low_pass_filter(G, s[0], s1)

##

s1 = s[k].copy()
s1[G.N//2] = 1.5*np.max(np.abs(s[k]))
high_pass_filter(G, s[k], s1, wc=20, order=1)

##

k = np.argmax([smoothness_and_gft(G, s[i])[0] for i in range(len(s))])

s1 = s[k] + np.random.normal(0, 3, size=G.N)
low_pass_filter(G, s[k], s1, wc=2, order=1)

# s1 = s[0] + np.random.normal(0, 1, size=G.N)
# low_pass_filter(G, s[0], s1)

##

k = np.argmax([smoothness_and_gft(G, s[i])[0] for i in range(len(s))])
high_pass_filter(G, s[k], s[k], wc=20, order=1)


##

k = np.argmin([smoothness_and_gft(G, s[i])[0] for i in range(len(s))])
compare_fourier_decomposition([G, G1, G2], [s[k], s[k], s[k]], titles=["Graphe de plus proche voisin", "Graphe de corrélation", "Graphe hybride"])






## Autres exemples (on s'en fout)

## Cas idéal

G = graphs.Logo()
G.compute_fourier_basis()
s = np.sum(G.U[:, :G.N//3], axis=1)
s1 = s + np.random.normal(0, 0.1*np.max(np.abs(s)), size=G.N)
low_pass_filter(G, s, s1, wc=4.2, order=5)

## Cas parfait (heureusement ça marche)

G = graphs.Logo()
G.compute_fourier_basis()
s = G.U[:, 2]
s1 = s + np.random.normal(0, 0.2*np.max(np.abs(s)), size=G.N)
low_pass_filter(G, s, s1, order=1)

## Détection d'anomalie (marche bien)

G = graphs.Logo()
G.compute_fourier_basis()
s = np.sum(G.U[:, :G.N//3], axis=1)
s1 = s.copy()
s1[G.N//2] = 1.5*np.max(np.abs(s))
high_pass_filter(G, s, s1, wc=20, order=1)

## Détection de 2 anomalies

G = graphs.Logo()
G.compute_fourier_basis()
s = np.sum(G.U[:, :G.N//3], axis=1)
s1 = s.copy()
s1[G.N//3] += 2*np.max(np.abs(s))
s1[2*G.N//3] += 2*np.max(np.abs(s))
high_pass_filter(G, s, s1, order=5)