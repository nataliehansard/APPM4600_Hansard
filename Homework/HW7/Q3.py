import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1/(1+(10*x)**2)

def weight(x_interp):
    N = len(x_interp)
    w = np.ones(N)
    for j in range(N):
        prod = 1
        for i in range(N):
            if i != j:
                prod *= (x_interp[j] - x_interp[i])
        w[j] = 1 / prod
    return w

def eval_phi(x_interp, x_eval):
    prod = np.ones_like(x_eval)
    for i in range(len(x_interp)):
        prod *= x_eval - x_interp[i]
    return prod

def barycentric(x_interp, y_interp, w, x_eval):
    sum = 0
    phi = eval_phi(x_interp, x_eval)
    for j in range(len(x_interp)):
        sum += ((w[j] / (x_eval - x_interp[j])) * y_interp[j])
    return phi*sum

for N in range(17, 18):  
    x_cheb = np.cos((2 * np.arange(1, N+1) - 1) * np.pi / (2 * N))
    y_cheb = f(x_cheb)

    w = weight(x_cheb)

    x_vals = np.linspace(-1, 1, 1001)
    y_vals = f(x_vals)
    y_bary = barycentric(x_cheb, y_cheb, w, x_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, 'k-', label="f(x)")
    plt.plot(x_vals, y_bary, '--', label=f"Barycentric Interpolation with Chebyshev Nodes(N={N})")
    plt.legend()
    plt.title(f"Barycentric/Chebyshev Interpolation for N={N}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    max_val = np.max(np.abs(y_bary))
    print(f"Max value for N={N}: {max_val}")

    if max_val > 100:
        break