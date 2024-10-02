import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import legendre

V = .001 
rho = 6.022e28 
k_B = 1.380649e-23 
theta = 428  

#gauss quadrature
N = 50
xi, wi = np.polynomial.legendre.leggauss(N)
I = 0
Ti = 50 #for part c

def integral(T):
    a = 0.
    b = theta / T
    
    func = lambda x: (x**4 * np.exp(x)) / (np.exp(x) - 1)**2

    x_map = 0.5 * (b-a) * (xi + 1) + .5 * (b+a) 
    w_map = 0.5 * (b-a) * wi        

    #sum wi*f(x)
    I = np.sum(w_map * func(x_map))
    return I

#part a:
def cv(T,N):
    coefficient = 9 * V * rho * k_B * (T / theta)**3
    I = integral(T)
    return coefficient * I



#plotting cv vs temp for part b:
temps = np.linspace(5, 500, 495)
cvs = np.array([cv(T,50) for T in temps])

Ns = np.arange(1, 80, 10) #for part c
cvs_partc = np.array([cv(Ti, n) for n in Ns])


plt.plot(temps, cvs, label="Heat Capacity $C(T)$")
plt.xlabel("temperature (K)")
plt.ylabel("c_v (J/K)")
plt.title("c_v vs temp")
plt.show()

plt.plot(Ns, cvs_partc, marker='o', label=f"c_v $C(T={Ti}K)$")
plt.xlabel("N")
plt.ylabel("c_v (J/K)")
plt.title(f"c_v at T={Ti}K vs N")
plt.show()
