import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fact

def H (n,x):
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    else:
        return (2*x*H(n-1,x) - 2*(n-1)*H(n-2,x))

def psi (n,x):
    coef = 1/(np.sqrt(2**n * fact(n) * np.sqrt(np.pi)))*np.exp(-x**2/2)
    return coef * H(n,x)

def psiGH (n,x):
    coef = 1/(np.sqrt(2**n * fact(n) * np.sqrt(np.pi)))#*np.exp(-x**2/2)
    return coef * H(n,x)
