import math

#value of x
x = 9.999999995000000e-10

#define function
y = math.e**x
unstable_result = y-1

#define true value
true_value = 1e-9

#compute the relative error
abs_error = abs(true_value - unstable_result)
rel_error = abs_error / true_value

#find num correct digits
print(unstable_result)
