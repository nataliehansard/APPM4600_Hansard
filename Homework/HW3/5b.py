import numpy as np
    
def fpi(f, x0, tol = 1e-10, nmax = 1000):
    x_n = x0

    for n in range(nmax):
        x_n1 = f(x_n)

        if abs(x_n1 - x_n) < tol:
            return x_n1, n
        x_n = x_n1 
    
    return None, nmax

def f(x):
    return -np.sin(2*x) + (5/4)*x - (3/4)

x0 = 2
root, n = fpi(f, x0)

if root is None:
    print("Method did not find root")
else:
    print(f"Root: {root}")
    print(f"Iterations: {n}")