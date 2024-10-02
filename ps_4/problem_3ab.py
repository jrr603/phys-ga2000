from hermite import psi
import numpy as np
import matplotlib.pyplot as plt


x_a = np.linspace(-4, 4, 1000)

for n in range(4):  # n = 0, 1, 2, 3
    psi_n = psi(n, x_a)
    plt.plot(x_a, psi_n, label=f'n = {n}')

plt.title('$\Psi_n$ for varying n')
plt.xlabel('x')
plt.ylabel('$\psi_n(x)$')
plt.legend()
plt.grid(True)
plt.show()

x_b = np.linspace(-10, 10, 500)
plt.plot(x_b,psi(30,x_b), label = "n=30")

plt.title('$\Psi_{30}$')
plt.xlabel('x')
plt.ylabel('$\psi_{30}(x)$')
plt.grid(True)
plt.show()