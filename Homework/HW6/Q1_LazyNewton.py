import numpy as np
from scipy.linalg import lu_factor, lu_solve

def F(x):
    return np.array([x[0]**2 + x[1]**2 - 4, np.exp(x[0]) + x[1] - 1])

def JF(x):
    return np.array([[2*x[0], 2*x[1]], [np.exp(x[0]), 1]])

def lazy_newton_method_nd(f, Jf, x0, tol, nmax, verb=False):
    xn = x0
    rn = x0.reshape(1, -1)
    Fn = f(xn)
    Jn = Jf(xn)
    lu, piv = lu_factor(Jn)
    n, nf, nJ, npn = 0, 1, 1, 1
    
    while npn > tol and n <= nmax:
        pn = -lu_solve((lu, piv), Fn)
        xn = xn + pn
        npn = np.linalg.norm(pn)
        rn = np.vstack((rn, xn))
        Fn = f(xn)
        nf += 1
        n += 1
    
    return (xn, rn, nf, nJ)

x0 = np.array([0.0, 0.0])
tol = 1e-6
nmax = 100
result = lazy_newton_method_nd(F, JF, x0, tol, nmax, True)
print("Root found by Lazy Newton:", result[0])
print("Number of Iterations:", result[2])