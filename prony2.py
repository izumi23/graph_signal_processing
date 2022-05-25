import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
plt.ion()
plt.show()

##

N = 20
lambdas = np.arange(N)
#h = (np.arange(N) <= 10)
h = lambdas
P, Q = 6, 4

a0 = np.transpose(np.tile(h, (P+1, 1)))
a1 = np.vander(lambdas, P+1, True) * a0
a2 = -np.vander(lambdas, Q+1, True)
A = np.concatenate((a1, a2), axis=1)
lb = -np.inf * np.ones((P+Q+2))
ub = np.inf * np.ones((P+Q+2))
lb[0] = 1 - 1e-14
ub[0] = 1 + 1e-14
b = np.zeros((N))

x = scipy.optimize.lsq_linear(A, b, bounds=(lb,ub)).x

p = np.flip(x[:P+1])
q = np.flip(x[P+2:])
f = lambda t: np.polyval(q, t) / np.polyval(p, t)

l = np.linspace(0, 20, 201)
plt.clf()
plt.plot(l, np.array([f(t) for t in l]))
plt.ylim([-0.1, 1.1])