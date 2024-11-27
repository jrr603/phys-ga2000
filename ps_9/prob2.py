import numpy as np
import matplotlib.pyplot as plt

def f1(stuff, t):
  rho = 1.22
  c = 0.47
  m = 1
  R = 0.08
  alpha = 9.8*np.pi*R**2 * rho * c / (2*m)
  x = stuff[0]
  vx= stuff[1]
  y = stuff[2]
  vy= stuff[3]
  f_x = vx #dx/dt
  f_vx = -alpha*vx*np.sqrt(vx**2+vy**2)
  f_y = vy #dx/dt
  f_vy = -1 -alpha*vy*np.sqrt(vx**2+vy**2)

  return np.array([f_x,f_vx,f_y,f_vy],float)
def f2(stuff, t):
  rho = 1.22
  c = 0.47
  m = 2
  R = 0.08
  alpha = 9.8*np.pi*R**2 * rho * c / (2*m)
  x = stuff[0]
  vx= stuff[1]
  y = stuff[2]
  vy= stuff[3]
  f_x = vx #dx/dt
  f_vx = -alpha*vx*np.sqrt(vx**2+vy**2)
  f_y = vy #dx/dt
  f_vy = -1 -alpha*vy*np.sqrt(vx**2+vy**2)

  return np.array([f_x,f_vx,f_y,f_vy],float)
def f3(stuff, t):
  rho = 1.22
  c = 0.47
  m = 3
  R = 0.08
  alpha = 9.8*np.pi*R**2 * rho * c / (2*m)
  x = stuff[0]
  vx= stuff[1]
  y = stuff[2]
  vy= stuff[3]
  f_x = vx #dx/dt
  f_vx = -alpha*vx*np.sqrt(vx**2+vy**2)
  f_y = vy #dx/dt
  f_vy = -1 -alpha*vy*np.sqrt(vx**2+vy**2)

  return np.array([f_x,f_vx,f_y,f_vy],float)
def f4(stuff, t):
  rho = 1.22
  c = 0.47
  m = 4
  R = 0.08
  alpha = 9.8*np.pi*R**2 * rho * c / (2*m)
  x = stuff[0]
  vx= stuff[1]
  y = stuff[2]
  vy= stuff[3]
  f_x = vx #dx/dt
  f_vx = -alpha*vx*np.sqrt(vx**2+vy**2)
  f_y = vy #dx/dt
  f_vy = -1 -alpha*vy*np.sqrt(vx**2+vy**2)

  return np.array([f_x,f_vx,f_y,f_vy],float)



def rk4_step(func, x, t, h):
  k1 = h*func(x,t)
  k2 = h*func(x+0.5*k1,t+0.5*h)
  k3 = h*func(x+0.5*k2,t+0.5*h)
  k4 = h*func(x+k3,t+h)
  return x + (k1+2*k2+2*k3+k4)/6


def rungekutta(func, x0, vx0, y0,vy0, h, t):
  tpoints = np.arange(0,t,h)
  x_list = []
  vx_list = []
  y_list = []
  vy_list = []
  stuff = np.array([x0,vx0,y0,vy0],float)

  for t in tpoints:
    if stuff[2] < 0:  # Stop if the projectile hits the ground
      break
    x_list.append(stuff[0])
    vx_list.append(stuff[1])
    y_list.append(stuff[2])
    vy_list.append(stuff[3])

    stuff = rk4_step(func, stuff, t, h)

  return tpoints, x_list, vx_list, y_list, vy_list
tpoints, x_list, vx_list, y_list, vy_list = rungekutta(f1, 0, 100*np.cos(np.pi/6),0, 100*np.sin(np.pi/6), .01, 1000)
plt.plot(x_list,y_list, label='m = 1')
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.title('Projectile Motion with air resistence')
plt.legend()
plt.show()

tpoints, x_list, vx_list, y_list, vy_list = rungekutta(f1, 0, 100*np.cos(np.pi/6),0, 100*np.sin(np.pi/6), .01, 1000)
plt.plot(x_list,y_list, label='m = 1')
tpoints, x_list, vx_list, y_list, vy_list = rungekutta(f2, 0, 100*np.cos(np.pi/6),0, 100*np.sin(np.pi/6), .01, 1000)
plt.plot(x_list,y_list, label = 'm = 2')
tpoints, x_list, vx_list, y_list, vy_list = rungekutta(f3, 0, 100*np.cos(np.pi/6),0, 100*np.sin(np.pi/6), .01, 1000)
plt.plot(x_list,y_list, label = 'm = 3')
tpoints, x_list, vx_list, y_list, vy_list = rungekutta(f4, 0, 100*np.cos(np.pi/6),0, 100*np.sin(np.pi/6), .01, 1000)
plt.plot(x_list,y_list, label= 'm = 4')
plt.title('Projectile Motion with different masses')
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.legend()
plt.show()