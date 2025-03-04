import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 


def driver():
    
    f = lambda x: math.exp(x)
    a = 0
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(a,b,Neval)
    
    ''' number of intervals'''
    Nint = 10
    
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,a,b,f,Nint)
    
    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
       fex(j) == f(xeval(j)) 
      
    plt.plot(xeval,fex)
    plt.plot(xeval,yeval)
    plt.show()   
     
    err = abs(yeval-fex)
    plt.plot(xeval,err)
    plt.show()            


def eval_lin_spline(x0, f_x0, x1, f_x1, alpha):
   
    if x0 == x1:
        return print("Error: x0 and x1 must be different")
    
    else: 
        f_alpha = f_x0 + (f_x1 - f_x0) * (alpha - x0) / (x1 - x0)
        return f_alpha
    

def  eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        
        '''temporarily store your info for creating a line in the interval of 
         interest'''
        a1= xint(j)
        fa1 = f(a1)
        b1 = xint(j+1)
        fb1 = f(b1)
        
        for kk in range(Nint):
           '''use your line evaluator to evaluate the lines at each of the points 
           in the interval'''
           '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
           the points (a1,fa1) and (b1,fb1)'''
           yeval(alpha(kk)) == eval_lin_spline(a1,fa1,b1,fb1,xeval(alpha(kk)))
           
x0, f_x0 = 1, 2
x1, f_x1 = 3, 4
alpha = 2
print(eval_lin_spline(x0, f_x0, x1, f_x1, alpha))

if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()               
