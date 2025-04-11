import numpy as np
from scipy.integrate import quad, simpson
from math import pi

def f(s):
    return 1 / (1 + s**2)

a, b = -5, 5

x_trap = np.linspace(a, b, 1291 + 1)
y_trap = f(x_trap)
Tn = (b - a) / 1291 * (0.5 * y_trap[0] + np.sum(y_trap[1:-1]) + 0.5 * y_trap[-1])

x_simp = np.linspace(a, b, 108 + 1)
y_simp = f(x_simp)
Sn = simpson(y_simp, x_simp)

val_default, _, info_default = quad(f, a, b, full_output=True)
evals_default = info_default['neval']

val_tol4, _, info_tol4 = quad(f, a, b, epsabs=1e-4, epsrel=1e-4, full_output=True)
evals_tol4 = info_tol4['neval']

print(f"Simpson (n=108):        {Sn:.8f}, Func Evals: {108+1}")
print(f"Trapezoidal (n=1291):   {Tn:.8f}, Func Evals: {1291+1}")
print(f"Quad (tol=1e-6):        {val_default:.8f}, Func Evals: {evals_default}")
print(f"Quad (tol=1e-4):        {val_tol4:.8f}, Func Evals: {evals_tol4}")