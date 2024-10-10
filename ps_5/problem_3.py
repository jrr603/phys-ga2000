import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
#print(os.getcwd()) 

data = pd.read_csv('/Users/jackson/Documents/GitHub/phys-ga2000/ps_5/signal.dat', delimiter='|')

data.columns = data.columns.str.strip() 

data = data[['time', 'signal']]

t_bad = data['time'].astype(float)
signal = data['signal'].astype(float)
t = (t_bad - t_bad.mean()) / t_bad.std()

plt.scatter(t_bad, signal, color='green')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('t vs Signal')
plt.show()

A = np.zeros((len(t), 4))
A[:, 0] = 1.
A[:, 1] = t 
A[:, 2] = t**2
A[:, 3] = t**3
#print(A.shape)
#print(A)

(u, w, vt) = np.linalg.svd(A, full_matrices=False)
#print(u.shape)
#print(w.shape)
#print(vt.shape)
#print(w)

ainv = vt.transpose().dot(np.diag(1. / w)).dot(u.transpose())
x = ainv.dot(signal)
bm = A.dot(x)

plt.plot(t, signal, '.', label='data')
plt.plot(t, bm, '.', label='model')
plt.xlabel('t')
plt.ylabel('b')
plt.legend()
plt.scatter(t, signal, color='green')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('t vs Signal')
plt.show()

plt.plot(bm, signal - bm, '.', label='data - model')
plt.xlabel('t')
plt.ylabel('Residuals (signal - fit)')
plt.legend()
plt.show()
