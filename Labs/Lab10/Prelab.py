import numpy as np

def eval_legendre(n, x):

    p = np.zeros(n+1)
    p[0] = 1
    if n > 0:
        p[1] = x
    
    for k in range(1, n):
        p[k+1] = ((2*k+1)*x*p[k]-k*p[k-1])/(k+1)
    
    return p

n = 5
x = 0.5
print(eval_legendre(n, x))
