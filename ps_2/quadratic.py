import numpy as np

def quadratic(a,b,c, dtype = np.float32):
    discriminant = (b**2 - 4*a*c) 
    if b > 0: #if b is positive 
        x1 = (-b - np.sqrt(discriminant))/(2*a)
        x2 = (2*c)/(-b - np.sqrt(discriminant))
    else:
        x1 = (2*c)/(-b + np.sqrt(discriminant))
        x2 = (-b + np.sqrt(discriminant))/(2*a)
    print(x1,x2)
    return(x2,x1)

