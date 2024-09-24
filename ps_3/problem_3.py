import numpy as np
import matplotlib.pyplot as plt

mu_Tl = np.log(2) / (3.053*60)  #decay constant from textbook: mu = ln2/tau

#Number of atoms at start
N = 1000

z = np.random.rand(N) #generating N random numbers
x = -np.log(1 - z) / mu_Tl #transforming our random numbers to our exponential decay distribution

sorted_times = np.sort(x) #sorting our times for efficiency 

#stuff to plot
t_values = np.linspace(0, max(sorted_times), 1000) #t
undecayed = [np.sum(sorted_times>t) for t in t_values]  #how many atoms not decayed at each t


plt.plot(t_values, undecayed, marker='o', label="Undecayed Atoms")
plt.xlabel('time (s)')
plt.ylabel('# undecayed atoms')
plt.title('# of undecayed atoms vs time')
plt.show()