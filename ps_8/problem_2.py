import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft,rfftfreq

#import pandas as pd
#piano = pd.read_csv('/Users/jackson/Documents/GitHub/phys-ga2000/ps_8/piano.txt', header=None).to_numpy().flatten()

piano = np.loadtxt('/Users/jackson/Documents/GitHub/phys-ga2000/ps_8/piano.txt')
trumpet = np.loadtxt('/Users/jackson/Documents/GitHub/phys-ga2000/ps_8/trumpet.txt')

plt.plot(np.arange(0, len(piano)), piano)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('Waveform of piano')
plt.grid()
plt.show()

plt.plot(np.arange(0, len(trumpet)), trumpet)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.title('Waveform of trumpet')
plt.grid()
plt.show()

def plot_fftcoefficients(data, title= 'yuh'):
    
    N = len(data)
    fft_values = rfft(data)
    freqs = rfftfreq(N, 1/44100)
    

    
    plt.plot(freqs[:10000], np.abs(fft_values[:10000]))  
    plt.title(f"{title} - Magnitudes of Fourier Coefficients")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()

plot_fftcoefficients(piano, "Piano")
plot_fftcoefficients(trumpet, "Trumpet")
