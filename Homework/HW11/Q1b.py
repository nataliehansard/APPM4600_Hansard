import numpy as np

def ddf(s):
    return (-2 + 6*s**2) / (1 + s**2)**3

s_vals = np.linspace(-5, 5, 10000)
max_fpp = np.max(np.abs(ddf(s_vals)))
print(f"Max |f''(s)| ≈ {max_fpp:.6f}")

def ddddf(s):
    return 24*(5*s**4 - 10*s**2 + 1) / (s**2 + 1)**5

s_vals = np.linspace(-5, 5, 10000)
max_fpppp = np.max(np.abs(ddddf(s_vals)))
print(f"Max |f''''(s)| ≈ {max_fpppp:.6f}")