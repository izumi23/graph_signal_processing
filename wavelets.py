import numpy as np
import matplotlib.pyplot as plt
import pygsp
from pygsp import graphs, filters, plotting

plt.ion()
plt.show()

##

def g(x):
    if x < 1:
        return x**2
    elif x <= 2:
        return -5 + 11*x - 6*x**2 + x**3
    else:
        return 4*x**-2

gamma = 1.384

def h(x, lmax):
    return gamma * np.exp(-( 10*x/(0.3*lmax) )**4)

def frequencies(lmax, r):
    return np.geomspace(lmax/40, lmax/2, r)

##

l = np.linspace(0, 10, 501)
lmax = 10
r = 7

plt.close('all')
fig, ax = plt.subplots()
ax.plot(l, np.array([h(x, lmax) for x in l]), label="passe-bas")
for w0 in frequencies(lmax, r):
    ax.plot(l, np.array([g(x/w0) for x in l]), label="{:.2f}".format(w0))
ax.legend()

##

