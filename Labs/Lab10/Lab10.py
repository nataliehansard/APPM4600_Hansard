import scipy 
from scipy.integrate import quad
import numpy as np

def eval_legendre(n, x):

    phi = np.zeros(n+1)
    phi[0] = 1
    if n > 0:
        phi[1] = x
    
    for k in range(1, n):
        phi[k+1] = ((2*k+1)*x*phi[k]-k*phi[k-1])/(k+1)
    
    return phi

def f(x):
    return 

def w(x): 
    return

def a(j):
    return quad(eval_legendre(n, x)*f(x)*w(x), -1, 1)/quad((eval_legendre(n, x))**2*w(x), -1, 1)

def p(x):
    return sum(eval_legendre(n, x)*a(j))

n = 5
x = 0.5
print(eval_legendre(n, x))
