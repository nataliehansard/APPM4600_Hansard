import numpy as np

# define systems of equations 
def F(x):
    f1 = x[0] + np.cos(x[0]*x[1]*x[2]) - 1
    f2 = (1-x[0])**(1/4) + x[1] + 0.05*x[2]**2 - 0.15*x[2] -1
    f3 = -x[0]**2 - 0.1*x[1]**2 + 0.01*x[1] + x[2] - 1
    return np.array([f1, f2, f3])

# define the jacobian
def JF(x):
    df1_dx = 1 -np.sin(x[0]*x[1]*x[2])*x[1]*x[2]
    df1_dy = -np.sin(x[0]*x[1]*x[2])*x[0]*x[2]
    df1_dz = -np.sin(x[0]*x[1]*x[2])*x[0]*x[1]
    df2_dx = -0.25*(1-x[0])**(-3/4)
    df2_dy = 1
    df2_dz = 0.1*x[2] - 0.15
    df3_dx = -2*x[0]
    df3_dy = -0.2*x[1] + 0.01
    df3_dz = 1
    return np.array([[df1_dx, df1_dy, df1_dz],
                     [df2_dx, df2_dy, df2_dz],
                     [df3_dx, df3_dy, df3_dz]])

def newtons_method(f, Jf, x0, tol, nmax, verb=False):
    xn = x0
    rn = np.array([xn])  # list of iterates
    n = 0
    npn = 1
    while(npn > tol and n <= nmax):
        Fn = f(xn)
        Jn = Jf(xn)
        pn = -np.linalg.solve(Jn, Fn)
        xn = xn + pn
        npn = np.linalg.norm(pn)
        n += 1
        rn = np.vstack((rn, xn))
    return xn, rn, n

def driver():
    x0 = np.array([-0.02, 0.09, 0.99])
    tol = 1e-6
    nmax = 100
    
    # Apply Newton's method
    rN, rnN, n = newtons_method(F, JF, x0, tol, nmax, verb=True)
    print("Root found by Newton's method:", rN)
    print("Number of Iterations:", n)

driver()