import numpy as np
import matplotlib.pyplot as plt

# define variables
R = 1.2
deltar = 0.1
f = 15
p = 0
theta = np.linspace(0, 2*np.pi, 1000)

# define x and y
x = R*(1 + deltar*np.sin(f*theta + p))*np.cos(theta)
y = R*(1 + deltar*np.sin(f*theta + p))*np.sin(theta)

# plot the curve
plt.figure(figsize=(8,8))
plt.plot(x, y)
plt.axis('equal')
plt.title('Parametric Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.show()