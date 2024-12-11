import numpy as np
from scipy.fft import rfft, irfft
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation  

hbar = 1.0546e-34  
L = 1e-8           
m = 9.109e-31      
N = 1000           
x = np.linspace(0, L, N)  
dx = L / N         
sigma = 1e-10      
kappa = 5e10       
dt = 1e-18        
tsteps = 2000     

def psi_initial(x):
    return np.exp(-((x - L / 2)**2) / (2 * sigma**2)) * np.exp(1j * kappa * x)

psi_0 = psi_initial(x)
psi_real = np.real(psi_0)
psi_imag = np.imag(psi_0)


k_vals = np.arange(0, N)  
E_k = (hbar**2 * (np.pi * k_vals / L)**2) / (2 * m)  

def dst(y):
    N = len(y)
    y2 = np.empty(2*N,float)
    y2[0] = y2[N] = 0.0
    y2[1:N] = y[1:]
    y2[:N:-1] = -y[1:]
    a = -np.imag(rfft(y2))[:N]
    a[0] = 0.0

    return a


def idst(a):
    N = len(a)
    c = np.empty(N+1,complex)
    c[0] = c[N] = 0.0
    c[1:N] = -1j*a[1:]
    y = irfft(c)[:N]
    y[0] = 0.0

    return y
    
# Calculate a_k and eta_k
a_k = dst(psi_real)
eta_k = -dst(psi_imag)


# Time evolution 
def psi_real_t(t):
    phase_cos = np.cos(E_k * t / hbar)  
    phase_sin = np.sin(E_k * t / hbar)  
    coeffs = a_k * phase_cos - eta_k * phase_sin
    
    return idst(coeffs)


PSI = np.zeros((tsteps, N))  
for i in range(tsteps):
    t = i * dt
    PSI[i, :] = psi_real_t(t)

# Pseudo animation
frames = 1
framestep = tsteps // frames

for i in range(frames):
    plt.plot(x, PSI[i * framestep, :])
    plt.ylim(-1, 1) 
    plt.xlabel('x (m)')
    plt.ylabel('Re($\Psi$)')
    plt.title(f'$\Psi$(x, t = {i * framestep * dt:.2e})')
    plt.show()


fig, ax = plt.subplots()
line, = ax.plot(x, PSI[0])
ax.set_xlabel('$x$ (m)')
ax.set_ylabel('$\psi(x)$ ')

def frame(i):
   line.set_data(x, PSI[i])
   return line,

ani = FuncAnimation(fig, frame, frames = 10000, interval=10, blit=True)
#ani.save('p2.gif', fps=30)
plt.show()