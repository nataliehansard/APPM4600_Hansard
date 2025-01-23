import matplotlib.pyplot as plt
import numpy as np

# define x values, delta, and zeta
x1 = np.pi
x2 = 10**6
delta = np.logspace(-16, 0, num=17)

# define the taylor function for x1 and x2
tay1 = (-delta*np.sin(x1)-((delta**2/2)*np.cos(x1)))
tay2 = (-delta*np.sin(x2)-((delta**2/2)*np.cos(x2)))
            
# define the function for original x1 and x2
orig1 = (np.cos(x1+delta)-np.cos(x1))
orig2 = (np.cos(x2+delta)-np.cos(x2))  

# define the error
y1 = np.abs(tay1-orig1)
y2 = np.abs(tay2-orig2)

# plot the two functions
plt.plot(delta, y1, label = f'x = {x1}')
plt.plot(delta, y2, label = f'x = {x2}')    

plt.yscale('log')
plt.xscale('log')
plt.ylabel('Error Approximation')
plt.xlabel('Delta (log scale)')
plt.title('Taylor Error Approximation')
plt.legend()
plt.show()