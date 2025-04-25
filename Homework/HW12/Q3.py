import numpy as np
from numpy.linalg import norm

# part A
def hilbert_matrix(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1 / (i + j + 1)
    return H

def power_method(A, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    q = np.random.rand(n)
    q = q / norm(q)
    eigenvalue = 0
    eigenvalues = []
    for i in range(max_iter):
        q_prev = q
        z = A @ q
        q = z / norm(z)
        eigenvalue_prev = eigenvalue
        eigenvalue = q @ A @ q
        eigenvalues.append(eigenvalue)
        if abs(eigenvalue - eigenvalue_prev) < tol:
            return eigenvalue, q, i + 1, eigenvalues
    return eigenvalue, q, max_iter, eigenvalues

iterations_needed = []
for n in range(4, 21, 4):
    H = hilbert_matrix(n)
    dominant_eigenvalue, dominant_eigenvector, iterations, _ = power_method(H)
    print(f"For n = {n}:")
    print(f"Dominant Eigenvalue = {dominant_eigenvalue:.8f}")
    print(f"Number of Iterations = {iterations}")
    iterations_needed.append((n, iterations))




# part b
n = 16
H = hilbert_matrix(n)
H_inv = np.linalg.inv(H)

smallest_eigenvalue_inv, smallest_eigenvector, iterations_inv, _ = power_method(H_inv)
smallest_eigenvalue = 1 / smallest_eigenvalue_inv

print(f"\nFor n = {n}:")
print(f"Smallest Eigenvalue = {smallest_eigenvalue:.8f}")
print(f"Number of Iterations (Inverse Power Method) = {iterations_inv}")

eigenvalues = np.linalg.eigvals(H)
smallest_eigenvalue_direct = np.min(np.abs(eigenvalues))
print(f"Smallest Eigenvalue (Direct Calculation) = {smallest_eigenvalue_direct:.8f}")
print(f"Relative Error = {abs(smallest_eigenvalue - smallest_eigenvalue_direct) / abs(smallest_eigenvalue_direct):.2e}")