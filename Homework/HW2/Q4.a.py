import math
import numpy as np
import matplotlib.pyplot as plt

# define t and y vectors
t = np.arange(0, np.pi + np.pi/30, np.pi/30)
y = np.cos(t)

# find the sum
S = np.sum(t*y)

print(f"the sum is: S = {S:.6f}")