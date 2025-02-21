import numpy as np
import matplotlib.pyplot as plt

def f(x,y,z):
    return x**2 + 4*y**2 + 4*z**2 - 16

def fx(x):
    return 2*x

def fy(y):
    return 8*y

def fz(z):
    return 8*z

def newtons_method(x0, y0, z0, tol=1e-6, nmax=100):
    x, y, z = x0, y0, z0
    n = 0
    errors = []
    n_num = []
    while (n < nmax):
        F = f(x, y, z)
        dfx = fx(x)
        dfy = fy(y)
        dfz = fz(z)
        d = F/(dfx**2 + dfy**2 + dfz**2)

        x_new, y_new, z_new = x - (d*dfx), y - (d*dfy), z - (d*dfz)
        error = np.sqrt((x_new - x)**2 + (y_new - y)**2 + (z_new - z)**2)
        errors.append(error)
        n_num.append(n)

        if np.abs(f(x_new, y_new, z_new)) < tol:
            return x_new, y_new, z_new, n+1, errors, n_num

        x, y, z = x_new, y_new, z_new
        n = n+1
    return x, y, z, n, n_num, errors

x0, y0, z0, = 1, 1, 1

x_newton, y_newton, z_newton, newton_count, n_num, errors = newtons_method(x0, y0, z0)
print(f"Newton's method solution: x = {x_newton}, y = {y_newton}, z = {z_newton} in {newton_count} iterations")

plt.plot(n_num, errors, marker='o')
plt.xlabel("Iteration number")
plt.ylabel("Error")
plt.title("Order of Convergence")
plt.show()