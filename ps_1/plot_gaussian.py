import numpy as np
import matplotlib.pyplot as plt

mu  = 0
std = 3

variance = np.square(std)

x = np.linspace(-10,10,1000)
f = np.exp(-np.square(x-mu)/(2*variance))/(np.sqrt(2*np.pi*variance))

plt.plot(x,f)
plt.xlabel('x')
plt.ylabel('y')
plt.title('gaussian')
plt.show()
