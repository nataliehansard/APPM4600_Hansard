import matplotlib.pyplot as plt
import numpy as np

Nmax = 100
tol = 1e-6

def p(tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count < Nmax):
       count = count +1
       p = np.linspace(0, 10, count)
       return[p]