import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
import sympy as sp

M = np.array([[1/6, 1/18],[0, 1/6]])

def F(x):
    return np.array([3*x[0]**2 - x[1]**2, 3*x[0]*x[1]**2 - x[0]**3 - 1])

def G(x):
    return x - M @ F(x)

def JF(x):
    return np.array([[6*x[0], -2*x[1]], [3*x[1]**2 - 3*x[0]**2, 6*x[0]*x[1]]])

def JG(x):
    return np.eye(2) - M @ JF(x)

def fixed_point_method_nd(G, JG, x0, tol, nmax, verb=False):
    xn = x0 
    rn = [x0]
    Gn = G(xn) 
    n = 0
    nf = 1 

    while np.linalg.norm(Gn - xn) > tol and n <= nmax:
        if verb:
            rhoGn = np.max(np.abs(np.linalg.eigvals(JG(xn))))
            print(f"Iteration {n}: x = {xn}, |G(x) - x| = {np.linalg.norm(Gn - xn):.2e}, Spectral Radius = {rhoGn:.2f}")

        xn = Gn 
        rn.append(xn) 
        Gn = G(xn) 
        n += 1
        nf += 1

        if np.linalg.norm(xn) > 1e15:
            n = nmax + 1
            nf = nmax + 1
            break

        if verb:
            if n >= nmax:
                print(f"Fixed-point iteration failed to converge, iterations = {nmax}, error = {np.linalg.norm(Gn - xn):.1e}\n")
            else:
                print(f"Fixed-point iteration converged, iterations = {n}, error = {np.linalg.norm(Gn - xn):.1e}\n")
        return np.array(xn), np.array(rn), n

x0 = np.array([1.0, 1.0])
tol = 1e-15
nmax = 1000

# Solve using Fixed-Point Iteration
solution, iterates, num_iterations = fixed_point_method_nd(G, JG, x0, tol, nmax, verb=True)

# Plot convergence
plt.plot(iterates[:, 0], iterates[:, 1], 'o-', label="Fixed-Point Iterates")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fixed-Point Iteration Convergence")
plt.legend()
plt.grid()
plt.show()

# Print final solution
print(f"Solution: x = {solution}")
print(f"Number of iterations: {num_iterations}")
print(f"Number of function evaluations: {num_iterations + 1}")