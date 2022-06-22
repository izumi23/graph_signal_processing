import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

plt.ion()
plt.show()
plt.rcParams['figure.autolayout'] = True

## Construction et calculs

def haar_basis_ordered(N):
    #les lignes de B forment une base de Haar en dimension N,
    #les vecteurs de base étant ordonnés par support décroissant
    N2 = 2 ** int(np.ceil(np.log2(N)))
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

def coefficients(B, s):
    #coefficients de la décomposition de s sur la base B
    return B @ s

def snr(s, s_compr):
    #signal-to-noise ratio (logarithmic)
    return -10 * np.log10( 1e-14 + np.mean((s_compr-s)**2) / np.mean(s**2) )

def compressed_snr(G, B, s):
    #on retire les composantes les plus faibles une par une
    #et on calcule le SNR à chaque étape
    s_hat = coefficients(B, s)
    order = np.argsort(np.abs(s_hat))
    snr_vect = np.zeros((G.N-1))
    s_compr = s.copy()
    for k in range(G.N-1):
        n = order[k]
        s_compr -= s_hat[n] * B[n]
        snr_vect[k] = snr(s, s_compr)
    return snr_vect

## Illustration

def show_haar_basis(G, B):
    fig = plt.figure(figsize=((12,6)))
    #affichage des 4 premiers éléments de la base
    gs = plt.GridSpec(2, 3)
    for i in range(4):
        ax = fig.add_subplot(gs[3*(i//2) + i%2])
        if i == 3:
            i = G.N - 2
        G.plot_signal(B[i+1], vertex_size=30, ax=ax)
        ax.set_title('$u_{{{}}}$'.format(i+2))
        ax.set_axis_off()
    #affichage 2D des N vecteurs de la base
    gs = plt.GridSpec(1, 3)
    ax = fig.add_subplot(gs[-1])
    im = ax.imshow(B)
    fig.colorbar(im, ax=ax, aspect=40)

def show_haar_basis2(G, B):
    fig, axes = plt.subplots(2, 3, figsize=((12,6)))
    axes = axes.flatten()
    ax = axes[0]
    G.compute_fourier_basis()
    G.plot_signal(G.U[:,1], vertex_size=30, ax=ax)
    ax.set_title("Deuxième vecteur propre")
    ax.set_axis_off()
    for i in range(1, 6):
        ax = axes[i]
        n = i if i < 4 else G.N-6+i
        G.plot_signal(B[n], vertex_size=30, ax=ax)
        ax.set_title('$u_{{{}}}$'.format(n+1))
        ax.set_axis_off()

def show_components(G, B, s, s_hat, nb_coef=5, suptitle=None):
    sh = np.abs(s_hat)
    comp = np.argsort(sh)
    fig = plt.figure(figsize=(4*((nb_coef+2)//2) ,8))
    #affichage du signal et de ses nb_coef composantes les plus
    #importantes
    gs = plt.GridSpec(3, (nb_coef+2)//2, height_ratios=[2, 2, 1])
    for k in range(nb_coef+1):
        n = comp[-k]
        ax = fig.add_subplot(gs[k])
        s1 = s if k==0 else B[n]
        G.plot_signal(s1, ax=ax, vertex_size=20)
        title = "Signal" if k==0 else "$\\hat s_{{{}}} = {:.3f}$".format(n, s_hat[n])
        ax.set_title(title)
        ax.set_axis_off()
    if suptitle is None:
        suptitle = "Composantes principales dans la base de Haar"
    fig.suptitle(suptitle)

    #affichage de chaque coefficient dans la base
    gs = plt.GridSpec(3, 1, height_ratios=[2, 2, 1])
    gs = gs[-1].subgridspec(1, 2)
    mk = '.' if G.N < 100 else 'None'
    ax = fig.add_subplot(gs[-2])
    ax.plot(np.arange(1, G.N+1), sh, linestyle='None', marker=mk)
    for n in range(G.N):
        ax.plot([n+1, n+1], [0, sh[n]], color='C0')
    ax.set_title("Coefficients dans la décomposition")
    #SNR en prenant uniquement les n composantes les plus importantes
    ax = fig.add_subplot(gs[-1])
    snr_vect = compressed_snr(G, B, s)
    ax.plot(np.arange(1, G.N), np.flip(snr_vect), marker=mk)
    ax.set_ylim(-2, 40)
    ax.set_title("SNR vs nombre de composantes gardées")

## Visualiser la base classique de Haar (ordonnée)

plt.close('all')
plt.imshow(haar_basis_ordered(21).transpose())
plt.colorbar()

## Visualiser la base de Haar d'un graphe

plt.close('all')
G = graphs.DavidSensorNet()
B = haar_basis(G)
show_haar_basis2(G, B)

## Exemple 1 : Dirac

G = graphs.DavidSensorNet()
s = np.zeros((G.N))
s[G.N//2] = 1
B = haar_basis(G)
s_hat = coefficients(B, s)
plt.close('all')
show_components(G, B, s, s_hat)

## Exemple 2 : Signal lisse

G = graphs.Logo()
s = np.array([np.sin(0.01*G.coords[i,0]) for i in range(G.N)])
s = s - np.average(s)
B = haar_basis(G)
s_hat = coefficients(B, s)
plt.close('all')
show_components(G, B, s, s_hat)

## Exemple 3 : Bretagne

H = np.loadtxt("data/GraphBretagneHybr.txt")
coords = np.loadtxt("data/GraphCoords.txt")
temp = np.loadtxt("data/Temperature.txt")
G = graphs.Graph(H)
G.set_coordinates(coords)

s = temp[0] - np.average(temp[0])
B = haar_basis(G)
s_hat = coefficients(B, s)
plt.close('all')
show_components(G, B, s, s_hat)

## Exemple 4 : Bruit

H = np.loadtxt("data/GraphBretagneHybr.txt")
coords = np.loadtxt("data/GraphCoords.txt")
G = graphs.Graph(H)
G.set_coordinates(coords)

s = np.random.normal(size=G.N)
B = haar_basis(G)
s_hat = coefficients(B, s)
plt.close('all')
show_components(G, B, s, s_hat)
