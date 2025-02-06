import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x - 4*np.sin(2*x) - 3

x = np.linspace(-2, 9, 400)
y = f(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) = x - 4sin(2x) - 3')
plt.grid(True)
plt.show()