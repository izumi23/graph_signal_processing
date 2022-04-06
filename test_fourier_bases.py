import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting

plt.ion()
plotting.BACKEND = 'matplotlib'
plt.rcParams['figure.figsize'] = (12, 8)


def show_fourier_basis(G):
    G.compute_fourier_basis()

    fig, axes = plt.subplots(2, 2)
    axes_flat = axes.flatten()
    for i, ax in enumerate(axes_flat):
        if i == len(axes_flat)-1:
            i = G.N - 2
        G.plot_signal(G.U[:, i+1], vertex_size=30, ax=ax)
        _ = ax.set_title('λ_{} = {:.3f}'.format(i+2, G.e[i+1]))
        ax.set_axis_off()

    fig.tight_layout()

show_fourier_basis(graphs.Logo())