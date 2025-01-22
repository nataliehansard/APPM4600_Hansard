import matplotlib.pyplot as plt
import numpy as np

# define x values and delta values
x1 = np.pi 
x2 = 10**6 
delta = np.logspace(-16, 0, num=17)

# define the function for x1
y1 = np.abs(np.cos(x1+delta)-np.cos(x1)+2*np.sin(x1+(delta/2))*np.sin(delta/2))

# define the function for x2
y2 = np.abs(np.cos(x2+delta)-np.cos(x2)+2*np.sin(x2+(delta/2))*np.sin(delta/2))


# plot the two function
plt.plot(delta, y1, label = f'x = {x1}')
plt.plot(delta, y2, label = f'x = {x2}')    

plt.yscale('log')
plt.xscale('log')
plt.ylabel('Difference')
plt.xlabel('Delta (log scale)')
plt.title('Difference between Cosine Expression and Cosine Approximation')
plt.legend()
plt.show()
