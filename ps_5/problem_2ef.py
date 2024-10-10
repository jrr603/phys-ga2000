import numpy as np

#using simpsons integration
def simpson(f,a,b,n):
    h=(b-a)/n
    k=0.0
    x=a + h
    for i in range(1,n//2 + 1):
        k += 4*f(x)
        x += 2*h

    x = a + 2*h
    for i in range(1,n//2):
        k += 2*f(x)
        x += 2*h
    return (h/3)*(f(a)+f(b)+k)
#write the integrand considering the transformation
def integrand(z, a): 
    c = a - 1
    x=z
    x = c*z/(1-z)
    dx_dz = (a - 1) / (1 - z) ** 2
    return np.exp((a - 1) * np.log(x) - x) * dx_dz

#evaluate integral
def gamma(a):
    I = simpson(lambda z: integrand (z,a), .0001, .999, 10000)
    return I


print(f"gamma(3/2) = {gamma(3/2)}") 
print(f"gamma(3) = {gamma(3)}\ngamma(6) = {gamma(6)}\ngamma(10) = {gamma(10)}")