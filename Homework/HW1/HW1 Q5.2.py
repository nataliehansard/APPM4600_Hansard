import matplotlib.pyplot as plt
import numpy as np

# define x values, delta, and zeta
x1 = np.pi
x2 = 10**6
delta = np.logspace(-16,0 , num=17)

# define the function for x1
y1 = np.abs(-delta*np.sin(x1)-((delta**2/2)*np.cos(x1)))

# define the function for x2
y2 = np.abs(-delta*np.sin(x2)-((delta**2/2)*np.cos(x2)))

# plot the two function
plt.plot(delta, y1, label = f'x = {x1}')
plt.plot(delta, y2, label = f'x = {x2}')    

plt.yscale('log')
plt.xscale('log')
plt.ylabel('Taylor Approximation')
plt.xlabel('Delta (log scale)')
plt.title('Taylor Approximation')
plt.legend()
plt.show()
