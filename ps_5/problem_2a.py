import numpy as np
import matplotlib.pyplot as plt
import math
import jax
import jax.numpy as j

def integrand(alpha, x):
    return x**(alpha-1) * np.exp(-x)

def integrand_jax(alpha, x):
    return x**(alpha-1) * jnp.exp(-x)
    
x_values = np.linspace(0, 5, 1000)
a_values = [2,3,4]
colors = ['blue', 'green', 'orange']
for i , a in enumerate(a_values):
    anal_values = [integrand(a, x) for x in x_values]
    #jax_values = [integrand_jax(a, x) for x in x_values]
    
    plt.plot(x_values, anal_values, label=f"(a={a})", color=colors[i], linestyle="-")
    #plt.plot(x_values, jax_values, label=f"jax Integral (a={a})", color=colors[i], linestyle=":")

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("$f(x) = x^{a-1} * e^{-x}$")
plt.legend()
plt.show()