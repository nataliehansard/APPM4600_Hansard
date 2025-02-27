import numpy as np
from numpy.linalg import inv 
from numpy.linalg import norm as np_norm

def driver():
    nmax = 100
    x0 = np.array([0.0, 0.0, 0.0])
    tol = 5e-2
    
    [root, gval, n] = steepest_descent(x0, tol, nmax)
    print("Steepest Descent Root:", root)
    print("G(root) = ", gval)
    print("Num Interations:", n)

def F(x):
    Fx = np.zeros(3)
    Fx[0] = x[0] + np.cos(x[0]*x[1]*x[2]) - 1
    Fx[1] = (1-x[0])**(1/4) + x[1] + 0.05*x[2]**2 - 0.15*x[2] -1
    Fx[2] = -x[0]**2 - 0.1*x[1]**2 + 0.01*x[1] + x[2] - 1
    return Fx

def JF(x):
    df1_dx = 1 -np.sin(x[0]*x[1]*x[2])*x[1]*x[2]
    df1_dy = -np.sin(x[0]*x[1]*x[2])*x[0]*x[2]
    df1_dz = -np.sin(x[0]*x[1]*x[2])*x[0]*x[1]
    df2_dx = -0.25*(1-x[0])**(-3/4)
    df2_dy = 1
    df2_dz = 0.1*x[2] - 0.15
    df3_dx = -2*x[0]
    df3_dy = -0.2*x[1] + 0.01
    df3_dz = 1
    return np.array([[df1_dx, df1_dy, df1_dz],
                     [df2_dx, df2_dy, df2_dz],
                     [df3_dx, df3_dy, df3_dz]])

def function_norm(x):
    Fx = F(x)
    return np.sum(Fx**2)

def gradient(x):
    Fx = F(x)
    J = JF(x) 
    return np.transpose(J) @ Fx

def steepest_descent(x, tol, nmax):
    for n in range(nmax):
        g1 = function_norm(x)
        z = gradient(x)
        z0 = np_norm(z)

        if z0 == 0:
            print("Gradient = Zero")
            return [x, g1, n+1]
            
        z = z/z0
        alpha3 = 1
        new_x = x - alpha3*z
        g3 = function_norm(new_x)

        while g3 >= g1:
            alpha3 = alpha3/2
            new_x = x - alpha3*z
            g3 = function_norm(new_x)
            
        if alpha3<tol:
            print("Descent has not improved")
            return [x,g1,n+1]
        
        alpha2 = alpha3/2
        new_x = x - alpha2*z
        g2 = function_norm(new_x)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        new_x= x - alpha0*z
        g0 = function_norm(new_x)

        if g0<=g3:
            alpha = alpha0
            gval = g0
        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            return [x,gval,n+1]

    print('Reached maximum number of iterations')      
    return [x,g1,n+1]

if __name__ == "__main__":
    driver()