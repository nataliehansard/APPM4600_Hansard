import math

#value of x
x = 9.999999995000000e-10

#define taylor series
def taylor_series(x, tol=10e-16):
    term = x
    sum_series = term
    n = 2

    while abs(term) > tol:
        term = (x**n) / math.factorial(n)
        sum_series += term
        n += 1
        print(n)
    return sum_series
taylor = taylor_series(x)

#define true value
true_value = 1e-9

#compute the relative error
abs_error = abs(true_value - taylor)
rel_error = abs_error / true_value

#find num correct digits
print(taylor)
print(rel_error)