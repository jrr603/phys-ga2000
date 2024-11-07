import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft, rfftfreq


#A) read dow and plot 
dow = np.loadtxt('/Users/jackson/Documents/GitHub/phys-ga2000/ps_8/dow.txt')
plt.plot(np.arange(0, len(dow)), dow)
plt.xlabel('days')
plt.ylabel('DOW closing value')
plt.title('dow data 2006 - 2010')
plt.grid()
plt.show()

#b calc fft coefficients
N = len(dow)
fft_coeffs = rfft(dow)
frequencies = rfftfreq(N)

print(dow.shape)
print(fft_coeffs.shape)

#reconstruct dow with only first 10 % coefs
fft_10 = fft_coeffs.copy()
fft_10[int(0.1 * len(fft_10)):] = 0  

#first 10% coefficients
data_10 = irfft(fft_10)

plt.plot(np.arange(0, len(dow)), dow, label='original data')
plt.plot(data_10, label='10 percent of Fourier Coefficients', linestyle='--',color='red')

plt.xlabel('days')
plt.ylabel('DOW closing value')
plt.title('dow data 2006 - 2010')
plt.grid()
plt.legend()

plt.show()


#now reconstruct with 2% coefs
fft_2 = fft_coeffs.copy()
fft_2[int(0.02 * len(fft_2)):] = 0  

data_2 = irfft(fft_2)

plt.plot(np.arange(0, len(dow)), dow,label='original data')
plt.plot(data_2, label='2 percent of Fourier Coefficients', linestyle='--',color='red')

plt.xlabel('days')
plt.ylabel('DOW closing value')
plt.title('dow data 2006 - 2010')
plt.legend()
plt.grid()
plt.show()
