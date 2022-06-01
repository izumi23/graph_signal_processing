import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
plt.ion()
plt.show()

##

def prony(vx, vy, P, Q, plot=False):
    """
    Approximation de moindres carrés vy*p(vx) = q(vx),
    où p et q sont des polynômes de degré P et Q respectivement,
    et p(0) = 1.
    """
    a0 = np.transpose(np.tile(vy, (P+1, 1)))
    a1 = np.vander(vx, P+1, True) * a0
    a2 = -np.vander(vx, Q+1, True)
    A = np.concatenate((a1, a2), axis=1)
    lb = -np.inf * np.ones((P+Q+2))
    ub = np.inf * np.ones((P+Q+2))
    lb[0] = 1 - 1e-14
    ub[0] = 1 + 1e-14
    lb[P+1] = 1 - 1e-14
    ub[P+1] = 1 + 1e-14
    b = np.zeros((len(vx)))

    sol_full = scipy.optimize.lsq_linear(A, b, bounds=(lb,ub))
    sol = sol_full.x
    p = np.flip(sol[:P+1])
    q = np.flip(sol[P+1:])

    if plot:
        f = lambda t: np.polyval(q, t) / np.polyval(p, t)
        l = np.linspace(np.min(vx)-0.1, np.max(vx)+0.1, 201)
        plt.clf()
        plt.plot(l, np.array([f(t) for t in l]))
        plt.plot(vx, vy, linestyle='None', marker='o')
        plt.ylim(-0.1, 1.1)

    return p, q

##

vx = np.arange(20)
vy = (vx <= 10) + 0.01  ## prony n'aime pas les zéros
P, Q = 6, 4
prony(vx, vy, P, Q, plot=True)