import numpy as np

A = np.array([[12, 10, 4],[10, 8, -5],[4, -5, 3]], dtype=float)

x = A[1:, 0] 

norm_x = np.linalg.norm(x)
sign = np.sign(x[0]) if x[0] != 0 else 1
v = x + sign * norm_x * np.array([1, 0])
v = v / np.linalg.norm(v)

H_ = np.eye(2) - 2*np.outer(v, v)

H = np.eye(3)
H[1:, 1:] = H_

M = H.T @ A @ H

M[np.abs(M) < 1e-12] = 0

print("Tridiagonal matrix M:")
print(np.round(M, 4))
