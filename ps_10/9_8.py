import numpy as np
import scipy.linalg
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation  
#newman excercize 9.8 
#uses crank nicholson to compute the time evolved wave function of an electron in a box. 

#parameters 
hbar = 1.0546e-36
L = 1e-8
m = 9.109e-31
N = 1000 # Grid slices
x0=L/2
sigma=1e-10 #meters 
kappa=5e10 #m^-1

#time step and spacial discretization
dt = 1e-18
dx = L/N

def cn_step(q=None, dt=None, dx=None ):
    #change alpha for the schrodinger equation
    alpha= dt * 1j * hbar / ( 4 * m * dx * dx)

    diag = (1. + 2 * alpha) * np.ones(len(q))
    offdiag = (- alpha) * np.ones(len(q))
    sparseA = np.zeros((3, len(q)), dtype=complex)


    sparseA[0, :] = offdiag
    sparseA[1, :] = diag
    sparseA[2, :] = offdiag
    qhalf = q 
    qhalf[1:-1] = q[1:-1] + alpha * (q[2:] - 2. * q[1:-1] + q[:-2])
    qnew = scipy.linalg.solve_banded((1, 1), sparseA, qhalf)
    return(qnew)

def psi(x): #wavefunction of traveling electron
  return np.exp(-(x-x0)**2/(2*sigma**2))*np.exp(1j*kappa*x)

#make spacial grid
xvals=np.linspace(0,L,N+1)

#initial wavefunction (t=0):
psi_0 = np.zeros(N+1, complex)
psi_0 = psi(xvals)
psi_0 = psi_0/np.linalg.norm(psi_0)
psi_0[0] = 0
psi_0[-1] = 0

#psi_1 = cn_step(psi_0, dt, dx)


#perform time steps:
tsteps=int(200000)
#build PSI array to store evolved values 
PSI = np.zeros((tsteps + 1, N + 1), dtype=complex)
PSI[0,:]=psi_0  


#crank nicholson evolution of wave function over time
for i in range(tsteps):
    PSI[i+1,:]=cn_step(PSI[i], dt, dx)


#pseudo animation, plotting snapshots of wavefunctions
frames=4

framestep=tsteps/frames
print(frames)
for i in range(frames):

    plt.plot ( xvals, PSI[i*int(framestep),:])
    #plt.xlim(0, 10)
    plt.ylim(-1, 1)
    plt.xlabel('x (m)')
    plt.ylabel('Re($\Psi $)')
    plt.title(f'$\Psi$(x, t = {i*framestep*dt:.2e})')
    plt.show()
