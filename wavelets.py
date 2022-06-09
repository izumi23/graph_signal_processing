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

def impulse_basis(G, r):
    #famille d'ondelettes issues de filtres, contenant
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
            B[G.N*l + node] = pygsp.filters.Filter(G, f).filter(d)
    return B

def haar_basis_ordered(N):
    #les lignes de B forment une base de Haar en dimension N,
    #les vecteurs de base étant ordonnés par support décroissant
    N2 = 2**(1 + int(np.log2(N)))
    B = [[0. for j in range(N)] for i in range(N2)]
    for j in range(N):
        B[0][j] = 1./np.sqrt(N)
    def dicho(a, b, i):
        if b-a >= 2:
            c = (a+b)//2
            for j in range(a, c):
                B[i][j] = np.sqrt((b-c)/(c-a)/(b-a))
            for j in range(c, b):
                B[i][j] = -np.sqrt((c-a)/(b-c)/(b-a))
            dicho(a, c, 2*i)
            dicho(c, b, 2*i+1)
    dicho(0, N, 1)
    i = 1
    while i < len(B):
        if max(B[i]) <= 1e-14:
            B.pop(i)
        else:
            i += 1
    return np.array(B)

def haar_basis(G):
    #on permute les colonnes de B pour que les sommets soient
    #classés selon leur valeur sur le 2ème vecteur propre
    G.compute_laplacian()
    G.compute_fourier_basis()
    s = G.U[:,1]
    nodes = np.argsort(s)
    B = haar_basis_ordered(G.N)
    return B[:, np.argsort(nodes)]

def coefficients(G, B, s):
    #coefficients sur toute la famille d'ondelettes
    return B @ s

def matching_pursuit(G, B, s, d):
    signal = s.copy()
    Base = np.zeros((d, G.N))
    Coefs = np.zeros((d+1, B.shape[0], B.shape[1]))
    for k in range(d):
        C = coefficients(G, B, signal)
        loc = np.argsort(C.flatten())
        i, node = loc[-1]//G.N, loc[-1]%G.N
        Coefs[k] = C
        Base[k] = B[i,node]
        signal = signal - C[i, node]/(B[i,node]@B[i,node])*B[i,node]
    Coefs[d] = coefficients(G, B, signal)
    return Base, Coefs, -10 * np.log10( np.mean((signal)**2) / np.mean(s**2))

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
        ax.set_title("")
        ax.set_axis_off()
    fig.suptitle("Ondelettes de " + wavelet_type + ", centrées en " + str(node))

def show_haar_basis(G, B):
    fig = plt.figure(figsize=((12,6)))
    gs = plt.GridSpec(2, 3)
    for i in range(4):
        ax = fig.add_subplot(gs[3*(i//2) + i%2])
        if i == 3:
            i = G.N - 2
        G.plot_signal(B[i+1], vertex_size=30, ax=ax)
        ax.set_title('$u_{{{}}}$'.format(i+2))
        ax.set_axis_off()
    gs = plt.GridSpec(1, 3)
    ax = fig.add_subplot(gs[-1])
    im = ax.imshow(B)
    fig.colorbar(im, ax=ax)

def show_components(G, B, C, s, suptitle=""):
    loc = np.argsort(np.abs(C))
    fig = plt.figure(figsize=(10,8))
    gs = plt.GridSpec(3, 2, height_ratios=[2, 2, 1])
    for k in range(4):
        ax = fig.add_subplot(gs[k])
        s1 = s if k==0 else B[loc[-k]]
        G.plot_signal(s1, ax=ax, vertex_size=20)
        l, node = loc[-k]//G.N, loc[-k]%G.N
        title = "Signal" if k==0 else "Ondelette "+str((l, node))
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(suptitle)
    gs = plt.GridSpec(3, 1, height_ratios=[2, 2, 1])
    ax = fig.add_subplot(gs[-1])
    im = ax.imshow(C.reshape(-1, G.N))
    fig.colorbar(im, ax=ax)

def show_matching_pursuit(G, B, s, d, suptitle=""):
    Base, Coefs, Snr = matching_pursuit(G, B, s, d)
    fig = plt.figure(figsize=(10,8))
    gs = plt.GridSpec(2*(d//2 + 1), 2, height_ratios=[2-k%2 for k in range(2*(d//2 + 1))])
    ax = fig.add_subplot(gs[0])
    G.plot_signal(s, ax=ax, vertex_size=20)
    title = "Signal"
    ax.set_title(title)
    ax.set_axis_off()
    for k in range(1,d+1):
        ax = fig.add_subplot(gs[k + 2*(k//2)])
        G.plot_signal(Base[k-1], ax=ax, vertex_size=20)
        title = "Ondelette "
        ax.set_title(title)
        ax.set_axis_off()

        s = (k%2 == 1)
        ax = fig.add_subplot(gs[k + 2*(k//2) + 2*s-1])
        im = ax.imshow(Coefs[k-1])
        fig.colorbar(im, ax=ax)
    fig.suptitle(suptitle)
    s = (d%2 == 0)
    ax = fig.add_subplot(gs[-1 -s])
    im = ax.imshow(Coefs[d])
    fig.colorbar(im, ax=ax)

    print("On obtient un SNR de {n:.2f}".format(n=Snr))

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

## Visualiser un extrait de la base impulsionnelle

plt.close('all')
G = graphs.DavidSensorNet()
B = impulse_basis(G, 5)
show_impulse_basis(G, B, G.N//2)

## Visualier la base de Haar

plt.close('all')
G = graphs.DavidSensorNet()
B = haar_basis(G)
show_haar_basis(G, B)

## Exemple 1 : Dirac

G = graphs.DavidSensorNet()
s = np.zeros((G.N))
s[G.N//2] = 1
B = impulse_basis(G, 5)
C = coefficients(G, B, s)
plt.close('all')
show_components(G, B, C, s)
#show_matching_pursuit(G, B, s, 3)

## Exemple 2 : Signal lisse

G = graphs.DavidSensorNet()
s = np.array([np.sin(G.coords[i,0]) for i in range(G.N)])
B = impulse_basis(G, 5)
C = coefficients(G, B, s)
plt.close('all')
show_components(G, B, C, s)
#show_matching_pursuit(G, B, s, 3)

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






