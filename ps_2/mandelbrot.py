import numpy as np
import matplotlib.pyplot as plt

N=1000
iterations = 100
r= np.linspace(-2,2,num =N, dtype=np.float32)
re,im = np.meshgrid(r,r)
c = re + 1j*im
z=np.zeros((N,N), dtype=complex) #initialize grid of zeros
fractal = np.full((N,N), True, dtype= np.bool)

#z' = z+c
for i in range(iterations):
   z[fractal] = z[fractal]*z[fractal] + c[fractal]
   fractal[np.abs(z)>2] = False



'''ChatGPT assisted the image generation. I used it to generate the plt.imshow() line'''
plt.figure(figsize=(10, 10))
plt.imshow(np.log(np.abs(z) + 1), cmap='inferno', extent=(-2, 2, -2, 2))
plt.title('Mandelbrot Set')
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xlabel('Real Componenet (c)')
plt.ylabel('Imaginary Component (c)')
plt.show()