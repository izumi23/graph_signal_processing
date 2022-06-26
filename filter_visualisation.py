import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

plt.rcParams['figure.autolayout'] = True

##

def show_filter_results(G, spectrum, s, s_hat, h, snr, discrete=False, suptitle=""):
    fig = plt.figure(figsize=(9, 9))
    fig.suptitle(suptitle)
    gs = plt.GridSpec(3, 2, width_ratios=[3, 2])
    titles = [
        "Signal d'origine",
        "Signal bruité (SNR = {:.1f})".format(snr[0]),
        "Signal filtré (SNR = {:.1f})".format(snr[1])
    ]
    for u in range(3):
        ax1 = fig.add_subplot(gs[2*u])
        G.plot_signal(s[u], ax=ax1, vertex_size=25, plot_name='')
        ax1.set_axis_off()

        ax2 = fig.add_subplot(gs[2*u+1])
        ax2.set_title(titles[u])
        ax2.plot(spectrum, np.abs(s_hat[u]), linestyle='None', marker='.')
        for freq, lambd in zip(spectrum, np.abs(s_hat[u])):
            ax2.plot([freq, freq], [0, lambd], color='C0')

        if u == 1:
            k = np.max(np.abs(s_hat))
            vx = spectrum if discrete else np.linspace(spectrum[0], spectrum[-1], 201)
            ax2.plot(vx, np.array([k*h(x) for x in vx]), color='C1', linewidth=2, label='Filtre')
            ax2.legend()
