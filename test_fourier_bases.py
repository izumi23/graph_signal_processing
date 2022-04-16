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
        if G.coords.shape[1] == 3 :
            ax = fig.add_subplot(gs[i], projection='3d')
        else : 
            ax = fig.add_subplot(gs[i])
        if i == nb_eig-1:
            i = G.N - 2
        G.plot_signal(G.U[:, i+1], vertex_size=30, ax=ax)
        _ = ax.set_title('λ_{} = {:.3f}'.format(i+2, G.e[i+1]))
        ax.set_axis_off()

    gs = plt.GridSpec(3, 1, height_ratios=[4, 4, 1])
    ax = fig.add_subplot(gs[-1])
    ax.axes.yaxis.set_ticks([])
    ax.set_ylim(0, 1)
    ax.vlines(G.e, 0, 1)


## Tests

show_fourier_basis(graphs.Logo())
show_fourier_basis(graphs.Ring(50))

# G = graphs.FullConnected(10)
# G.set_coordinates()
# show_fourier_basis(G)

#show_fourier_basis(graphs.DavidSensorNet())

## Ring avec une arête vers le centre

N = 50
W = np.eye(N+1,N+1,-1) + np.eye(N+1,N+1,1)
W[-2,0],W[0,-2] = 1,1
W[-2,-1],W[-1,-2] = 0,0
W[0,-1],W[-1,0] = 1,1
G = graphs.Graph(W)

A = np.array([[1+np.cos(2*k*np.pi/N), 1+np.sin(2*k*np.pi/N)] for k in range(N)] + [[1,1]])

G.set_coordinates(A)
show_fourier_basis(G)

## FullConnected avec une arête en moins

N = 10
W = np.ones((N+1,N+1)) - np.eye(N+1)
W[0,1], W[1,0] = 0,0
print(W)
G = graphs.Graph(W)

G.set_coordinates()
show_fourier_basis(G)

## Tore

M,N = 20,20
W = np.eye(M*N,M*N,-1) + np.eye(M*N,M*N,1) + np.eye(M*N,M*N,M) + np.eye(M*N,M*N,-M) + np.eye(M*N,M*N,M*N-M) + np.eye(M*N,M*N,-M*N+M)
for k in range(N):
    W[k*M-1,k*M], W[k*M,k*M-1] = 0,0
    W[k*M+M-1,k*M],W[k*M, k*M+M-1] = 1,1
A = np.array([[(1 + np.cos(2*i*np.pi/M)/3)*np.cos(2*k*np.pi/N), (1 + np.cos(2*i*np.pi/M)/3)*np.sin(2*k*np.pi/N), np.sin(2*i*np.pi/M)/3] for i in range(M) for k in range(N)])
G = graphs.Graph(W)

G.set_coordinates(A)
show_fourier_basis(G)






