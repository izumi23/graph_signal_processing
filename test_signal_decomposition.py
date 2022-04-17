import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting
from numpy import pi, cos, sin

plt.ion()
plotting.BACKEND = 'matplotlib'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.autolayout'] = True


def fourier_decomposition(G, s, ax1, ax2):
    G.plot_signal(s, ax=ax1, vertex_size=30, plot_name='')
    ax1.set_axis_off()
    G.compute_fourier_basis()
    s_hat = G.gft(s)
    smoothness = (s @ G.L @ s) / (s @ s)
    label = 'λ_x = {:.3f}'.format(smoothness)
    ax2.axvline(smoothness, linewidth=2, color='C1', label=label)
    ax2.plot(G.e, np.abs(s_hat), linestyle='None', marker='.')
    for i in range(G.N):
        ax2.plot([G.e[i], G.e[i]], [0, np.abs(s_hat[i])], color='C0')
    ax2.legend()

def show_fourier_decomposition(G, s, leave_open=False):
    if not leave_open:
        plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fourier_decomposition(G, s, axes[0], axes[1])

def compare_fourier_decomposition(vG, vs):
    plt.close('all')
    nb_sig = len(vG)
    assert(nb_sig == len(vs))
    fig, axes = plt.subplots(nb_sig, 2, figsize=(9, 3*nb_sig))
    for u in range(nb_sig):
        fourier_decomposition(vG[u], vs[u], axes[u][0], axes[u][1])


## Test 1

G = graphs.DavidSensorNet()

s1 = np.zeros((G.N), dtype=float)
for i in range(G.N):
    x, y = G.coords[i]
    if y < 0.66:
        s1[i] = 1 if x < 0.66 else 2
    else:
        s1[i] = 3 if x < 0.4 else 4

s2 = np.array([G.coords[i][0] for i in range(G.N)])
s3 = np.array([sin(10*(G.coords[i][0])) for i in range(G.N)])

compare_fourier_decomposition([G, G, G], [s1, s2, s3])


## Test 2

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















