import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

##

W = np.loadtxt("data/GraphBretagneHybr.txt")
coords = np.loadtxt("data/GraphCoords.txt")
temp = np.loadtxt("data/Temperature.txt")
G = graphs.Graph(W)
G.set_coordinates(coords)
G.compute_fourier_basis()
spectrum_sample = np.loadtxt("data/spectrum_sample.txt")
N = 43

##

def show_signal(G, spectrum, s, s_hat):
    fig = plt.figure(figsize=(9, 3))
    gs = plt.GridSpec(1, 2, width_ratios=[3, 2])

    ax1 = fig.add_subplot(gs[0])
    G.plot_signal(s, ax=ax1, vertex_size=25, plot_name='')
    ax1.set_axis_off()

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(spectrum, np.abs(s_hat), linestyle='None', marker='.')
    for freq, val in zip(spectrum, np.abs(s_hat)):
        ax2.plot([freq, freq], [0, val], color='C0')

def fit_into_interval(s, a, b):
    m, M = np.min(s), np.max(s)
    return (b-a)/(M-m)*(s-m) + a

##

plt.close('all')
k = 1
s_hat = spectrum_sample[:N*k:k]
s = fit_into_interval(G.igft(s_hat), -2, 5)
s_hat = G.gft(s)
show_signal(G, G.e, s, s_hat)

##

a, b = -2, 5
temp = np.zeros((2, N))
s_hat = spectrum_sample[:N]
temp[0] = fit_into_interval(G.igft(s_hat), a, b) #signal lisse
s_hat = spectrum_sample[:2*N:2]
temp[1] = fit_into_interval(G.igft(s_hat), a, b) #signal très lisse
np.savetxt("data/TemperatureGenerated.txt", temp, fmt='%.1f')
