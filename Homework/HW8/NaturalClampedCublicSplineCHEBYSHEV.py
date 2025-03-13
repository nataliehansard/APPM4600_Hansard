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
    # Generate Chebyshev nodes
    x_nodes = np.zeros(n)
    for j in range(n):
        x_nodes[j] = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2*j + 1) * np.pi / (2 * n))
    
    x_nodes = np.sort(x_nodes)
   
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
    plt.title(f'Natural vs Clamped (with Chebyshev Nodes) N = {n}')
    plt.legend()

    plt.figure() 
    err_N = abs(y_spline_natural-y_val)
    err_C = abs(y_spline_clamped-y_val)
    plt.semilogy(x_val,err_N,'ro--',label='Natural')
    plt.semilogy(x_val,err_C,'bs--',label='Clamped')
    plt.title(f'Error of Natural vs Clamped Splines (with Chebyshev Nodes) for N = {n}')
    plt.legend()
    plt.show()
    
plt.tight_layout()
plt.show()