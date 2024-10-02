import numpy as np
from scipy.special import roots_hermite
from hermite import psiGH

def integral_gh(n, N=100):
    xi, wi = roots_hermite(N)

    func = lambda x: (x**2) * psiGH(n,x)**2

    I = np.sum(wi * func(xi))
    return I

def uncertainty(n):
    x2_expectation = integral_gh(n)
    return np.sqrt(x2_expectation)

n_value = 5
exact = np.sqrt(n_value+0.5)
uncertainty_value = uncertainty(n_value)
print(f"Uncertainty (√⟨x^2⟩) for n = {n_value} is approximately: {uncertainty_value}")

print(f"is this exact?\n\nexact value = {exact}\nmy value = {uncertainty_value}\nexact - my value = {exact - uncertainty_value}")
