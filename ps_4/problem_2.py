import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import legendre

N = 20
xi, wi = np.polynomial.legendre.leggauss(N)

def integral(amp):
    a = 0.  
    b = amp  
    
    def func(x):
        y = amp**4 - x**4
        return 1 / np.sqrt(y) if y > 0 else 0  #avoid 0 in deniminator 
    
    x_map = 0.5 * (b - a) * (xi + 1) + 0.5 * (b + a) #map from -1,1 to a,b
    w_map = 0.5 * (b - a) * wi        
    
    y = amp**4 - x_map**4 #
    valid = np.where(y > 0, y, np.inf)  #replace bad values with inf to avoid 0 in denominator
    func_values = 1 / np.sqrt(valid)
    
    I = np.sum(w_map * func_values)
    return I

def period(a):
    coefficient = np.sqrt(8)
    I = integral(a)
    return coefficient * I

amplitudes = np.linspace(0.01, 2, 100) 
Ts = np.array([period(a) for a in amplitudes])

plt.plot(amplitudes, Ts)
plt.xlabel("Amplitude")
plt.ylabel("Period (s)")
plt.title("Period vs Amplitude")
plt.show()
