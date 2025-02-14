import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(3*x) - 27*(x**6) + 27*(x**4)*np.exp(x) - 9*(x**2)*np.exp(2*x)

def fprime(x):
    return 3*np.exp(3*x) - (27*6)*(x**5) + (27*4)*(x**3)*np.exp(x) + 27*(x**4)*np.exp(x) - 18*x*np.exp(2*x) - 36*(x**2)*np.exp(2*x)

x_val = np.linspace(3, 5, 100)
f_val = [f(x) for x in x_val]
plt.plot(x_val, f_val, label = 'f(x)')
plt.title('Graph of f(x)')
plt.axhline(0, color = 'black', linestyle = '--')
plt.xlabel('x values')
plt.ylabel('f(x)')
plt.legend()
plt.show()

def newton(x0, tol=1e-13, nmax=100):
    for n in range(nmax):
        fx = f(x0)
        fprimex = fprime(x0)
        if abs(fx) < tol:
            return x0
        if fprimex == 0:
            print('Derivative is zero, Newton fails')
            return None
        x1 = x0 - (fx/fprimex)
    return x1

def secant(x0, x1, tol=1e-13, nmax=100):
    for n in range(nmax):
        fx0 = f(x0)
        fx1 = f(x1)
        if abs(fx1) < tol:
            return x1
        x2 = x1 - fx1 * ((x1 - x0)/(fx1 - fx0))
        x0, x1 = x1, x2
    return x2

def scalar(x0, m, tol=1e-13, nmax=100):
    for n in range(nmax):
        fx0 = f(x0)
        fprimex = fprime(x0)
        if abs(fx0) < tol:
            return x0
        x1 = x0 - m*(fx0/fprimex)
    return x1


guess1 = 3.0
guess2 = 5.0
m = 3
new = newton(guess1)
sec = secant(guess1, guess2)
scal = scalar(guess1, m)
print(f"Newton Method Rood: {new:.6f}")
print(f"Secant Method Rood: {sec:.6f}")
print(f"Scalar Multiple Method Rood: {scal:.6f}")