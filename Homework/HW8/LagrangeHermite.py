import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

def driver():

    f = lambda x: 1/(1+x**2)
    fp = lambda x: -2*x/(1+x**2)**2

    N = 20
    ''' interval'''
    a = -5
    b = 5
   
    ''' create equispaced interpolation nodes'''
    xint = np.linspace(a,b,N+1)
    
    ''' create  interpolation data'''
    yint = np.zeros(N+1)
    ypint = np.zeros(N+1)
    for jj in range(N+1):
        yint[jj] = f(xint[jj])
        ypint[jj] = fp(xint[jj])
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yevalL = np.zeros(Neval+1)
    yevalH = np.zeros(Neval+1)
    for kk in range(Neval+1):
      yevalL[kk] = eval_lagrange(xeval[kk],xint,yint,N)
      yevalH[kk] = eval_hermite(xeval[kk],xint,yint,ypint,N)

    ''' create vector with exact values'''
    fex = np.zeros(Neval+1)
    for kk in range(Neval+1):
        fex[kk] = f(xeval[kk])
    
    
    plt.figure()
    plt.plot(xeval,fex,'ro-', label='True Function')
    plt.plot(xeval,yevalL,'bs--',label='Lagrange') 
    plt.plot(xeval,yevalH,'c.--',label='Hermite')
    plt.title(f'Lagrange vs Hermite vs True Function for N = {N}')
    plt.semilogy()
    plt.legend()
    plt.show()

    plt.figure() 
    err_L = abs(yevalL-fex)
    err_H = abs(yevalH-fex)
    plt.semilogy(xeval,err_L,'ro--',label='Lagrange')
    plt.semilogy(xeval,err_H,'bs--',label='Hermite')
    plt.title(f'Error of Langrange vs Hermite for N = {N}')
    plt.legend()
    plt.show()
                   
def eval_hermite(xeval,xint,yint,ypint,N):

    ''' Evaluate all Lagrange polynomials'''
    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
              

    yeval = 0.
    
    for jj in range(N+1):
       Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
       Rj = (xeval-xint[jj])*lj[jj]**2
       yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
       
    return(yeval)
       

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
       
if __name__ == '__main__':
  driver()        
