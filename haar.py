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

## Illustration

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
    fig.colorbar(im, ax=ax, aspect=40)

def show_components(G, B, s, s_hat, nb_coef=5, suptitle=""):
    sh = np.abs(s_hat)
    comp = np.argsort(sh)
    fig = plt.figure(figsize=(4*((nb_coef+2)//2) ,8))
    gs = plt.GridSpec(3, (nb_coef+2)//2, height_ratios=[2, 2, 1])
    for k in range(nb_coef+1):
        n = comp[-k]
        ax = fig.add_subplot(gs[k])
        s1 = s if k==0 else B[n]
        G.plot_signal(s1, ax=ax, vertex_size=20)
        title = "Signal" if k==0 else "$\\hat s_{{{}}} = {:.3f}$".format(n, s_hat[n])
        ax.set_title(title)
        ax.set_axis_off()
    fig.suptitle(suptitle)
    gs = plt.GridSpec(3, 1, height_ratios=[2, 2, 1])
    ax = fig.add_subplot(gs[-1])
    ax.plot(np.arange(1, G.N+1), sh, linestyle='None', marker='.')
    for n in range(G.N):
        ax.plot([n+1, n+1], [0, sh[n]], color='C0')

## Visualiser la base classique de Haar (ordonnée)

plt.close('all')
plt.imshow(haar_basis_ordered(21))
plt.colorbar()

## Visualiser la base de Haar d'un graphe

plt.close('all')
G = graphs.DavidSensorNet()
B = haar_basis(G)
show_haar_basis(G, B)

## Exemple 1 : Dirac

G = graphs.DavidSensorNet()
s = np.zeros((G.N))
s[G.N//2] = 1
B = haar_basis(G)
s_hat = coefficients(B, s)
plt.close('all')
show_components(G, B, s, s_hat)

## Exemple 2 : Signal lisse

G = graphs.DavidSensorNet()
s = np.array([np.sin(G.coords[i,0]) for i in range(G.N)])
s = s - np.average(s)
B = haar_basis(G)
s_hat = coefficients(B, s)
plt.close('all')
show_components(G, B, s, s_hat)
