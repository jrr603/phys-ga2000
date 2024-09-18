 #quadratic equation
import numpy as np

def quadratic_(a,b,c, dtype = np.float32):
   # x = np.roots(np.array([a,b,c]))
   #return(x)
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
      return("no real roots")
    x1 = (-b + np.sqrt(discriminant))/(2*a)
    x2 = (-b - np.sqrt(discriminant))/(2*a)
    x1_ = (2*c)/(-b + np.sqrt(discriminant))
    x2_ = (2*c)/(-b - np.sqrt(discriminant))
    return(F"roots:\nx1 = {x1}\nx2 = {x2}\nroots from part b:\nx1 ={x1_}\nx2 ={x2_}")
    
print(quadratic_(.001,1000,.001))

#part c????
