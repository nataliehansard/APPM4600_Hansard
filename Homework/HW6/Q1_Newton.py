import numpy as np

# define systems of equations 
def F(x):
    fxy = x[0]**2 + x[1]**2 - 4 
    gxy = np.exp(x[0]) + x[1] - 1 
    return np.array([fxy, gxy])

# define the jacobian
def JF(x):
    df_dx = 2*x[0]
    df_dy = 2*x[1]
    dg_dx = np.exp(x[0])
    dg_dy = 1
    return np.array([[df_dx, df_dy], [dg_dx, dg_dy]])

def newtons_method(f, Jf, x0, tol, nmax, verb=False):
    xn = x0
    Fn = f(xn)
    rn = np.array([xn])  # list of iterates
    n = 0
    npn = 1
    while(npn > tol and n <= nmax):
        Jn = Jf(xn)
        pn = -np.linalg.solve(Jn, Fn)
        xn = xn + pn
        npn = np.linalg.norm(pn)
        n += 1
        rn = np.vstack((rn, xn))
        Fn = f(xn)
    return xn, rn, n

def driver():
    x0 = np.array([0.0, 0.0])  # Initial guess
    tol = 1e-6
    nmax = 100

    # Apply Newton's method
    rN, rnN, n = newtons_method(F, JF, x0, tol, nmax, verb=True)
    print("Root found by Newton's method:", rN)
    print("Number of Iterations:", n)

driver()