import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import brentq

def f(x):
    return x**6 - x - 1

def fprime(x):
    return 6*x**5 - 1

alpha = opt.brentq(f, 1, 2)
print(f"Exact root (alpha): {alpha:.6f}")

def newton(x0, alpha, tol=1e-6, nmax=100):
    errors = []
    x = x0
    for n in range(nmax):
        fx = f(x)
        fprimex = fprime(x)
        if abs(fx) < tol:
            break
        if fprimex == 0:
            print('Derivative is zero, Newton fails')
            return None
        xn = x - (fx/fprimex)
        errors.append(abs(xn - alpha))
        x = xn
    return x, errors

def secant(x0, x1, alpha, tol=1e-6, nmax=100):
    errors = []
    for n in range(nmax):
        fx0 = f(x0)
        fx1 = f(x1)
        if abs(fx1) < tol:
            break
        x2 = x1 - fx1 * ((x1 - x0)/(fx1 - fx0))
        errors.append(abs(x2 - alpha))
        x0, x1 = x1, x2
    return x2, errors

guess1 = 2.0
guess2 = 1.0

new, errors_newton = newton(guess1, alpha)
sec, errors_secant = secant(guess1, guess2, alpha)

print("Iteration | Newton Error | Secant Error")
for i in range(min(len(errors_newton), len(errors_secant))):
    print(f"{i+1:^10}|{errors_newton[i]:^13.8f} | {errors_secant[i]:^13.8f}")

plt.loglog(errors_newton[:-1], errors_newton[1:], 'o-', label= 'Newton')
plt.loglog(errors_secant[:-1], errors_secant[1:], '*-', label= 'Secant')
plt.xlabel("x_k - alpha")
plt.ylabel("x_(k+1) - alpha")
plt.legend()
plt.title("Convergence Error: Newton vs Secant")
plt.show()

newton_slope = np.polyfit(np.log(errors_newton[:-1]), np.log(errors_newton[1:]), 1)[0]
secant_slope = np.polyfit(np.log(errors_secant[:-1]), np.log(errors_secant[1:]), 1)[0]

print(f"Newton's Error Slope: {newton_slope:.6f}")
print(f"Secant's Error Slope: {secant_slope:.6f}")