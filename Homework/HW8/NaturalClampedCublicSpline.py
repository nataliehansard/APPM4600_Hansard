import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def f(x):
    return 1 / (1 + x**2)

def df(x):
    return -2*x / (1 + x**2)**2

n_values = [5, 10, 15, 20]
a, b = -5, 5
x_val = np.linspace(a, b, 1000)
y_val = f(x_val)

plt.figure(figsize=(10, 8))

for i, n in enumerate(n_values):
    x_nodes = np.linspace(a, b, n+1)
    y_nodes = f(x_nodes)
    dy_nodes = df(x_nodes)
    
    # Natural Cubic Spline (second derivative at a = 0 and b = 0)
    spline_natural = CubicSpline(x_nodes, y_nodes, bc_type=((2, 0), (2, 0)))
    y_spline_natural = spline_natural(x_val)
    
    # Clamped Cubic Spline (first derivative at df(a) = 0 and df(b) = 0)
    spline_clamped = CubicSpline(x_nodes, y_nodes, bc_type=((1, df(x_nodes[0])), (1, df(x_nodes[-1]))))
    y_spline_clamped = spline_clamped(x_val)
    
    plt.subplot(2, 2, i+1) 
    plt.plot(x_val, y_val, 'k-', label='True Function')
    plt.plot(x_val, y_spline_natural, 'b--', label='Natural Spline')
    plt.plot(x_val, y_spline_clamped, 'm-', label='Clamped Spline')
    plt.title(f'n = {n}')
    plt.legend()
    
plt.tight_layout()
plt.show()
