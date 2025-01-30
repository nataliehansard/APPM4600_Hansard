import random
import numpy as np
import matplotlib.pyplot as plt

# define for loop and variables
for i in range(1, 11):
    R = i
    deltar = 0.05
    f = 2 + i
    p = random.uniform(0, 2)
    theta = np.linspace(0, 2*np.pi, 1000)

    # define x and y
    x = R*(1 + deltar*np.sin(f*theta + p))*np.cos(theta)
    y = R*(1 + deltar*np.sin(f*theta + p))*np.sin(theta)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('10 Parametric Curves')
    plt.axis('equal')

# configure the plot
plt.figure(figsize=(8,8))
plt.show()