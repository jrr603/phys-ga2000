import numpy as np
import matplotlib.pyplot as plt

m = 1000
N = np.arange(10,1010,10)

#where to store stuff: mean, var, skew, kurt
ymeans = []
yvars  = []
yskews = []
ykurts = [] 

for n in N:
  x = np.random.exponential(scale = 1. ,size = (m,n))
  y  = np.mean(x, axis = 1)

  mean = np.mean(y)
  var = np.var(y)
  skew = np.mean((y - mean)**3) / (np.std(y)**3)
  kurtosis = np.mean((y - mean)**4) / (np.std(y)**4)

  ymeans.append(mean)
  yvars.append(var)
  yskews.append(skew)
  ykurts.append(kurtosis)

  if n == 1000: #create a histogram at some large N to seee if gaussian visually
    plt.hist(y, bins=69, color='red', density = True) #nice
    plt.title(f'Distribution of y for N={n}')
    plt.xlabel('y')
    plt.ylabel('')
     
    #creating a gaussian plot over my histogram  
    mu = mean 
    std = np.sqrt(var)

    y_values = np.linspace(min(y), max(y), 100)
    gaussian = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y_values - mu) / std)**2)
    plt.plot(y_values, gaussian, color='black')

    plt.show()

#Plot 4 more graphs!
plt.plot(N,ymeans)
plt.title('Mean vs N')
plt.xlabel('N')
plt.ylabel('mean')
plt.show()

plt.plot(N,yvars)
plt.title('Variance vs N')
plt.xlabel('N')
plt.ylabel('variance')
plt.show()

plt.plot(N,yskews)
plt.title('Skew vs N')
plt.xlabel('N')
plt.ylabel('skew')
plt.show()

plt.plot(N,ykurts)
plt.title('Kurtosis vs N')
plt.xlabel('N')
plt.ylabel('kurtosis')
plt.show()