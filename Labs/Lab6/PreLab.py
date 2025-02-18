import numpy as np

def f(s):
    return cos(s)

def centeredfprime(s):
    return (f(s+h)-f(s-h))/(2*h)

def forwardfprime(s):
    return (f(s+h)-f(s))/(h)

s = np.pi/4
h = 0.01*2.00**(-np.arange(0,10))

