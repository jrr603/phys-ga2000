
import matplotlib.pyplot as plt
import numpy as np

###Runge-Kutta 4####
def f1(x,t) :
  # Simple Harmonic Oscillator
  omega = 1
  position = x[0]
  velocity= x[1]
  f_position = velocity #dx/dt
  f_velocity = -omega**2 * position #dv/dx = d2x/dx2
  return np.array([f_position,f_velocity],float)

def f2(x,t):
  omega = 1
  position = x[0]
  velocity= x[1]
  f_position = velocity #dx/dt
  f_velocity = -omega**2 * position**3 #dv/dx = d2x/dx2
  return np.array([f_position,f_velocity],float)

def f3(x, t):
  mu =1
  omega = 1
  position = x[0]
  velocity= x[1]
  f_position = velocity #dx/dt
  f_velocity = -omega**2*position + mu*(1-position**2)*velocity
  return np.array([f_position,f_velocity],float)

def rk4_step(func, x, t, h):
  k1 = h*func(x,t)
  k2 = h*func(x+0.5*k1,t+0.5*h)
  k3 = h*func(x+0.5*k2,t+0.5*h)
  k4 = h*func(x+k3,t+h)
  return x + (k1+2*k2+2*k3+k4)/6


def rungekutta(func, x0, y0, h, t):
  tpoints = np.arange(0,t,h)
  positions_list = []
  velocities_list = []
  x = np.array([x0,y0],float)

  for t in tpoints:
    positions_list.append(x[0])
    velocities_list.append(x[1])
    x = rk4_step(func, x, t, h)

  return tpoints, positions_list, velocities_list

tpoints, positions_list, velocities_list = rungekutta(f3, 1, 0, 0.01, 20)
plt.plot(positions_list, velocities_list,label=' mu = 4')

'''
tpoints, positions_list, velocities_list = rungekutta(f2, 2, 0, 0.01, 50)
plt.plot(tpoints,positions_list,label=' x(0) = 2')

tpoints, positions_list, velocities_list = rungekutta(f2, 0.25, 0, 0.01, 50)
plt.plot(tpoints,positions_list,label=' x(0) = 0.25')
tpoints, positions_list, velocities_list = rungekutta(f2, 0.5, 0, 0.01, 50)
plt.plot(tpoints,positions_list,label=' x(0) = 0.5')

#tpoints1, positions_list1, velocities_list1 = rungekutta(f, 0.25, 0, 0.01, 50)
#plt.plot(tpoints,positions_list1,label=' Harmonic Oscillator')
'''
plt.legend()
plt.title('Van Der Pol Oscillator Phase-Space Diagram')
plt.xlabel('position(m)')
plt.ylabel('velocity (m/s)')
plt.show()