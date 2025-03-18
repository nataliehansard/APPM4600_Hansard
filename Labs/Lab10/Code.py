import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad

def driver():

    f = lambda x: 1/(1+x**2)

    a = -1
    b = 1

    w = lambda x: 1.

    n = 2

    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
        pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])

    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])

    plt.figure()
    plt.plot(xeval,fex,'ro-', label= 'f(x)')
    plt.plot(xeval,pval,'bs--',label= 'Expansion')
    plt.legend()
    plt.show()
    
    err = abs(pval-fex)
    plt.semilogy(xeval,err,'ro--',label='error')
    plt.legend()
    plt.show()

def eval_legendre(n, x):

    p = np.zeros(n+1)
    p[0] = 1
    if n > 0:
        p[1] = x
    
    for k in range(1, n):
        p[k+1] = ((2*k+1)*x*p[k]-k*p[k-1])/(k+1)
    
    return p

def legendre( j, x):
    if j == 0:
        return 1
    elif j == 1:
        return x
    else:
        return eval_legendre(j,x)[-1]

def eval_legendre_expansion(f,a,b,w,n,x):
    p = eval_legendre(n,x)
    pval = 0.0

    for j in range(0,n+1):

        phi_j = lambda x: legendre( j, x)

        phi_j_sq = lambda x: phi_j(x)**2*w(x)

        norm_fac,err = quad(phi_j_sq,a,b)

        func_j = lambda x: phi_j(x)*f(x)*w(x)/norm_fac

        aj,err = quad(func_j,a,b)

        pval = pval+aj*p[j]
    
    return pval

if __name__ == '__main__':

    driver()