import numpy as np

def f(x,y):
    return 3*x**2 - y**2

def g(x,y):
    return 3*x*y**2 - x**3 - 1

def iteration_scheme(x0, y0, tol=1e-6, nmax=100):
    M = np.array([[1/6, 1/18], [0, 1/6]])
    x, y = x0, y0
    n = 0
    while(n < nmax):
        F = np.array([f(x,y), g(x,y)])
        p = M * F
        x_new, y_new = x - p[0], y - p[1]

        if np.linalg.norm(p) < tol:
            return x_new, y_new, n+1

        x, y = x_new, y_new
        n = n+1
    return x[1], y[1], n

def jacobian(x, y):
    return np.array([[6*x, -2*y], [3*y**2 - 3*x**2, 6*x*y]])

def newton_method(x0, y0, tol=1e-6, Nmax=100):
    x, y = x0, y0
    N = 0
    while (N < Nmax):
        F = np.array([f(x, y), g(x, y)])
        J = jacobian(x, y)
        p = np.linalg.solve(J, -F)
        x_new, y_new = x + p[0], y + p[1]

        if np.linalg.norm(p) < tol:
            return x_new, y_new, N+1

        x, y = x_new, y_new
    return x, y, Nmax

x0, y0 = 1, 1

x_n, y_n, n_count = iteration_scheme(x0, y0)
print(f"Iteration scheme solution: x = {x_n[1]}, y = {y_n[1]} in {n_count} iterations")

x_newton, y_newton, newton_count = newton_method(x0, y0)
print(f"Newton's method solution: x = {x_newton}, y = {y_newton} in {newton_count} iterations")
