import numpy as np
from scipy.integrate import simpson

def g(t):
    return t * np.cos(1 / t)

a, b = 1e-6, 1 
n = 4 

t_vals = np.linspace(a, b, n + 1)
g_vals = g(t_vals)

I_approx = -simpson(g_vals, t_vals)
print(f"Simpsons Approximation ≈ {I_approx:.8f}")
