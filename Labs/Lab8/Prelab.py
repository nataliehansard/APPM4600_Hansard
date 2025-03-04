def eval_lin_spline(x0, f_x0, x1, f_x1, alpha):
   
    if x0 == x1:
        return print("Error: x0 and x1 must be different")
    
    else: 
        f_alpha = f_x0 + (f_x1 - f_x0) * (alpha - x0) / (x1 - x0)
        return f_alpha

x0, f_x0 = 1, 2
x1, f_x1 = 3, 4
alpha = 2
print(eval_lin_spline(x0, f_x0, x1, f_x1, alpha))