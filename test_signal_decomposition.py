import numpy as np
import matplotlib.pyplot as plt
import pygsp
import contextily as cx
from pygsp import graphs, filters, plotting
from numpy import pi, cos, sin

plt.ion()
plotting.BACKEND = 'matplotlib'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.autolayout'] = True


def smoothness_and_gft(G, s):
    G.compute_fourier_basis()
    s_hat = G.gft(s)
    i = 0
    while abs(G.e[i]) < 1e-14:
        s_hat[i] = 0
        i += 1
    s0 = G.igft(s_hat)
    return (s0 @ G.L @ s0) / (s0 @ s0), s_hat

def fourier_decomposition(G, s, ax1, ax2, h=None, show_map=False):
    G.plot_signal(s, ax=ax1, vertex_size=15, plot_name='')
    ax1.set_axis_off()
    if show_map:
        cx.add_basemap(ax, crs=open("data/Map.txt").read(), zoom=8)
    smoothness, s_hat = smoothness_and_gft(G, s)
    #label = 'λ_x = {:.3f}'.format(smoothness)
    #ax2.axvline(smoothness, linewidth=2, color='C1', label=label)
    ax2.set_title('ν(x) = {:.3f}'.format(smoothness))
    ax2.plot(G.e, np.abs(s_hat), linestyle='None', marker='.')
    for i in range(G.N):
        ax2.plot([G.e[i], G.e[i]], [0, np.abs(s_hat[i])], color='C0')
    if h is not None:
        vx = np.linspace(0, G.e[-1], 201)
        k = np.max(np.abs(s_hat))
        ax2.plot(vx, np.array([k*h(x) for x in vx]), color='C1', linewidth=2, label='Filtre')
    #ax2.legend()

def show_fourier_decomposition(G, s, leave_open=False):
    if not leave_open:
        plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fourier_decomposition(G, s, axes[0], axes[1])

def compare_fourier_decomposition(vG, vs, h=None, leave_open=False, suptitle="", titles=None):
    if not leave_open:
        plt.close('all')
    nb_sig = len(vG)
    assert(nb_sig == len(vs))
    fig = plt.figure(figsize=(9, 3*nb_sig))
    gs = plt.GridSpec(nb_sig, 2, width_ratios=[3, 2])
    for u in range(nb_sig):
        ax1 = fig.add_subplot(gs[2*u])
        ax2 = fig.add_subplot(gs[2*u+1])
        fourier_decomposition(vG[u], vs[u], ax1, ax2, h if u==1 else None)
        if titles is not None:
            ax1.set_title(titles[u])
    fig.suptitle(suptitle)

def compare_fourier_decomposition_horizontal(vG, vs, h=None, leave_open=False, suptitle="", titles=None):
    if not leave_open:
        plt.close('all')
    nb_sig = len(vG)
    assert(nb_sig == len(vs))
    fig = plt.figure(figsize=(5*nb_sig, 9))
    gs = plt.GridSpec(2, nb_sig)
    for u in range(nb_sig):
        ax1 = fig.add_subplot(gs[u])
        ax2 = fig.add_subplot(gs[nb_sig+u])
        fourier_decomposition(vG[u], vs[u], ax1, ax2, h if u==1 else None)
    fig.suptitle(title)


## Tests

def test1():
    G = graphs.DavidSensorNet()

    s1 = np.zeros((G.N), dtype=float)
    for i in range(G.N):
        x, y = G.coords[i]
        if y < 0.66:
            s1[i] = 1 if x < 0.66 else 2
        else:
            s1[i] = 3 if x < 0.4 else 4

    # s3 = np.array([G.coords[i][0] for i in range(G.N)])
    s3 = np.arange(G.N)
    s2 = np.array([sin(10*(G.coords[i][0])) for i in range(G.N)])

    compare_fourier_decomposition_horizontal([G, G, G], [s1, s2, s3])


def test2():
    G = graphs.Ring(50)

    G1 = graphs.Ring(50)
    G1.W[0, 25] = G1.W[25, 0] = 1
    G1.Ne += 1
    G1.compute_laplacian()

    G2 = graphs.Ring(50)
    for i in range(-2, 3):
        G2.W[i, 25-i] = G2.W[25-i, i] = 1
        G2.Ne += 1
    G2.compute_laplacian()

    s = np.cos(np.arange(50)*2*pi/50)

    compare_fourier_decomposition([G, G1, G2], [s, s, s])

def show_bretagne(show_map=False, show_edges=True, title=""):
    W = np.loadtxt("data/GraphBretagne.txt")
    coords = np.loadtxt("data/GraphCoords.txt")
    s = np.loadtxt("data/Temperature.txt")
    G = graphs.Graph(W)
    G.set_coordinates(coords)

    fig, ax = plt.subplots()
    G.plot_signal(s[0], ax=ax, show_edges=show_edges, plot_name=title)
    if show_map:
        cx.add_basemap(ax, crs=open("data/Map.txt").read(), zoom=8)
    ax.set_axis_off()

def gft_bretagne():
    W = np.loadtxt("data/GraphBretagne.txt")
    coords = np.loadtxt("data/GraphCoords.txt")
    s = np.loadtxt("data/Temperature.txt")
    G = graphs.Graph(W)
    G.set_coordinates(coords)

    kmin = np.argmin([smoothness_and_gft(G, s[i])[0] for i in range(len(s))])
    kmax = np.argmax([smoothness_and_gft(G, s[i])[0] for i in range(len(s))])
    compare_fourier_decomposition([G, G], [s[kmin], s[kmax]])

