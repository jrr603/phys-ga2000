#### Problem 3 on PS2

import numpy as np
import timeit

def madelung_for(L):
    v = 0.
    time_start = timeit.default_timer()
    for i in range(-L,L):
        for j in range (-L,L):
            for k in range (-L,L):
                R = np.sqrt(i*i + j*j + k*k)
                if R != 0:
                    v += (pow(-1,(i+j+k))) * 1/R 
    time_end = timeit.default_timer()
    elapsed = time_end - time_start
    return(v,elapsed)

size, t = madelung_for(50)
print("Mandelung Constant with for loop: ", size)
print("Elapsed time with for loop: ", t)

def madelung_mesh(L):
  timestart_mesh = timeit.default_timer()
  r = np.arange(-L,L, dtype = np.float64)
  i,j,k = np.meshgrid(r,r,r) #create lattice
  R = np.sqrt(i*i + j*j + k*k)

  V = (-1) ** (i+j+k) * 1 *R
  V=1/V[R!=0]
  
  madelung = np.sum(V)
  timeend = timeit.default_timer()
  elapsed =  timeend - timestart_mesh 
  return(madelung,elapsed)

size, t = madelung_mesh(50)
print("Mandelung Constant with mesh: ", size)
print("Elapsed time with mesh: ", t)