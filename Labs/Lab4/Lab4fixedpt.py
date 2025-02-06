# import libraries
import numpy as np
    
def driver():

# test functions 
     g = lambda x: (10/(x+4))**0.5

     Nmax = 100
     tol = 1e-6

# test g '''
     x0 = 1.5
     [xstar,ier,x] = fixedpt(g,x0,tol,Nmax)
     print('the approximate fixed point is:',xstar)
     print('g(xstar):',g(xstar))
     print('Error message reads:',ier)
    

# define routines
def fixedpt(g,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    x=[x0];
    count = 0
    while (count <Nmax):
       count = count +1
       x1 = g(x0)
       x=np.append(x,[x1])
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, x]

print(x)


driver()