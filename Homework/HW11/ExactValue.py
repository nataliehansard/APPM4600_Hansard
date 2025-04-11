from math import atan, pi

a, b = -5, 5
exact_val = atan(b) - atan(a)
print(f"Exact value using antiderivative: {exact_val:.15f}")