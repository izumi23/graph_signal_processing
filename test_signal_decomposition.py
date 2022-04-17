import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

plt.ion()
plotting.BACKEND = 'matplotlib'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.autolayout'] = True


def fourier_decomposition(G, s, ax1, ax2):
    G.plot_signal(s, ax=ax1, vertex_size=30, plot_name='')
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
    fig = plt.figure(figsize=(9, 3*nb_sig))
    for u in range(nb_sig):
        if G.coords.shape[1] == 3 :
            ax1 = fig.add_subplot(nb_sig, 2, 2*u+1, projection='3d')
        else : 
            ax1 = fig.add_subplot(nb_sig, 2, 2*u+1)
        ax2 = fig.add_subplot(nb_sig, 2, 2*u+2)
        fourier_decomposition(vG[u], vs[u], ax1, ax2)


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
s3 = np.array([np.sin(10*(G.coords[i][0])) for i in range(G.N)])

compare_fourier_decomposition([G, G, G], [s1, s2, s3])


## Test 2



















