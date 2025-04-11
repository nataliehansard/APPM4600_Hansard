import numpy as np
from scipy.integrate import quad, simpson, trapezoid

def f(s):
    return 1 / (1 + s**2)

a, b = -5, 5

x = np.linspace(a, b, 101) 
y = f(x)

trap_result = trapezoid(y, x)
simp_result = simpson(y, x)
print(f"Trapezoidal result: {trap_result:.6f}")
print(f"Simpson result:   {simp_result:.6f}")