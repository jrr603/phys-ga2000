import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import curve_fit
#print(os.getcwd()) 

data = pd.read_csv('/Users/jackson/Documents/GitHub/phys-ga2000/ps_5/signal.dat', delimiter='|')

data.columns = data.columns.str.strip() 

data = data[['time', 'signal']]

t_bad = data['time'].astype(float)
signal = data['signal'].astype(float)
t = (t_bad - t_bad.mean()) / t_bad.std()

order = 30

A = np.zeros((len(t), order+1))
for n in range(order+1):
    
    A[:, n] = t**n
print(A.shape)
print(A)

(u, w, vt) = np.linalg.svd(A, full_matrices=False)
#print(u.shape)
#print(w.shape)
#print(vt.shape)
#print(w)

ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
x = ainv.dot(signal)
bm = A.dot(x)

#plt.plot(t, signal, '.', label='data')
plt.plot(t, bm, '.', label='model', color = "black")
plt.xlabel('t')
plt.ylabel('b')
plt.legend()
plt.scatter(t, signal, color='green')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('t vs Signal')
plt.show()

residuals = signal - bm
plt.plot(t, residuals, '.', label='data - model', color='red')
plt.xlabel('t')
plt.ylabel('Residuals (signal - fit)')
plt.legend()
plt.show()
condition_number = np.max(w) / np.min(w)
print(w)
print(f'Condition number of the design matrix: {condition_number}')
