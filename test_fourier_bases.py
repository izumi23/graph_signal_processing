import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

plt.ion()
plotting.BACKEND = 'matplotlib'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.autolayout'] = True


def show_fourier_basis(G):
    plt.close('all')
    nb_eig = 4
    G.compute_fourier_basis()

    fig = plt.figure()
    gs = plt.GridSpec(3, 2, height_ratios=[4, 4, 1])

    for i in range(nb_eig):
        ax = fig.add_subplot(gs[i])
        if i == nb_eig-1:
            i = G.N - 2
        G.plot_signal(G.U[:, i+1], vertex_size=30, ax=ax)
        _ = ax.set_title('λ_{} = {:.3f}'.format(i+2, G.e[i+1]))
        ax.set_axis_off()

    gs = plt.GridSpec(3, 1, height_ratios=[4, 4, 1])
    ax = fig.add_subplot(gs[-1])
    ax.plot(G.e, np.zeros_like(G.e), linestyle='None', marker='.', markersize=1)


## Tests

#show_fourier_basis(graphs.Logo())
#show_fourier_basis(graphs.Ring(50))

G = graphs.FullConnected(10)
G.set_coordinates()
show_fourier_basis(G)

show_fourier_basis(graphs.DavidSensorNet())











