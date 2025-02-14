import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

def f(x, alpha = 0.138e-6, t = 60*24*60*60, Ti = 20, Ts = -15):
    return (Ti-Ts)*sp.erf(x/(2*np.sqrt(alpha*t))) + Ts

def fprime(x, alpha = 0.138e-6, t = 60*24*60*60, Ti = 20, Ts = -15):
    term = x / (2 * np.sqrt(alpha * t))
    return (35/np.sqrt(np.pi*alpha*t))*np.exp((-term**2)/(2*np.sqrt(alpha*t)))

x_val = np.linspace(0, 10, 100)
f_val = [f(x) for x in x_val]
plt.plot(x_val, f_val, label = 'f(x)')
plt.title('Temperature Distribution in Soil')
plt.axhline(0, color = 'black', linestyle = '--')
plt.xlabel('Depth (meters)')
plt.ylabel('f(x)')
plt.legend()
plt.show()

def bisection(f, a, b, nmax=100, tol=1e-13):
    if (f(a)*f(b) > 0):
        print('No root found')
        return None
    if (f(a)== 0):
        print('Root found:', a)
        return a
    if (f(b) == 0):
        print('Root found:', b)
        return b
    
    n = 0
    while (n < nmax and (b-a)/2 > tol):
        c = (a+b)/2
        if f(c) == 0:
            return c
        elif (f(a)*f(c) < 0):
            b = c
            n = n+1
        else:
            a = c
            n = n+1
        return (a+b)/2

def newton(x0, tol=1e-13, max=100):
    for m in range(max):
        fx = f(x0)
        fprimex = fprime(x0)
        if abs(fx) < tol:
            return x0
        if fprimex == 0:
            print('Derivative is zero, Newton fails')
            return None
        x0 = x0 - (fx/fprimex)
    return x0

a0 = 0
b0 = 2
xguess = 0.01
bi = bisection(f, a0, b0)
new = newton(xguess)
print(f"Bisection Method Depth: {bi:.13f} m")
print(f"Newton Method Depth: {new: .13f} m")