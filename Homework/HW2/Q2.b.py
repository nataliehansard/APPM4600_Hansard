import numpy as np

# define A and A inverse
A = np.array([[1, 1], [1+1e-10, 1-1e-10]]) * 0.5
Ainv = np.array([[1-1e10, 1e10], [1+1e10, -1e10]])

# find the 2 norms
A_2norm = np.linalg.norm(A, 2)
Ainv_2norm = np.linalg.norm(Ainv, 2)

print(f"A 2norm = {A_2norm}")
print(f"A inverse 2norm = {Ainv_2norm}")

# find the condition number
condition = A_2norm * Ainv_2norm
print(f"Condition number = {condition}")
