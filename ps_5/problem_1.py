import numpy as np
import matplotlib.pyplot as plt
import math
import jax
import jax.numpy as jnp

def f(x):
  return 1 + 0.5*math.tanh(2*x) 

def derf(x):
  h = 1e-5
  return (f(x+h) - f(x-h)) / (2*h)

def anal_derf(x):
  return (1-math.tanh(2*x)**2)

def f_jax(x):
    return 1 + 0.5 * jnp.tanh(2 * x)

x_values = np.linspace(-2, 2, 300)

analderf_values = [anal_derf(x) for x in x_values]
derf_values = [derf(x) for x in x_values]

jax_derf = jax.grad(f_jax)

jax_derf_values = [jax_derf(x) for x in x_values]


plt.plot(x_values, analderf_values, label="Analytic Derivative", color="yellow", linestyle="-")
plt.plot(x_values, derf_values, label="Central Difference Derivative", color="black", linestyle=":")
plt.xlabel("x")
plt.ylabel("f '(x)")
plt.title("Derivative of f(x) = 1 + 0.5*tanh(2x)")
plt.legend()
plt.show()

plt.plot(x_values, jax_derf_values, label="jax Derivative", color="red", linestyle="-")
plt.xlabel("x")
plt.ylabel("f '(x)")
plt.title("Derivative of f(x) = 1 + 0.5*tanh(2x)")
plt.legend()
plt.show()