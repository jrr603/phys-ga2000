import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import legendre
from hermite import psi

#gauss quadrature
N = 100
xi, wi = np.polynomial.legendre.leggauss(N)

def integral(n):
    nn = n
    func = lambda x: (x**2) * np.abs(psi(nn,x))**2

    #x_map = 0.5 * (b-a) * (xi + 1) + .5 * (b+a) 
    #w_map = 0.5 * (b-a) * wi        
    x_map = np.tan(np.pi / 2 * xi)  # Transform points
    w_map = (np.pi / 2) * (1 / np.cos(np.pi / 2 * xi)**2) * wi  # Adjust weights
    #sum wi*f(x)
    I = np.sum(w_map * func(x_map))
    return I

n_value = 5
x2_expectation = integral(n_value)

uncertainty = np.sqrt(x2_expectation)
print(f"Uncertainty (sqrt⟨x^2⟩) for n = {n_value} is approximately: {uncertainty}")
