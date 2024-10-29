import jax.numpy as jnp
from jax import grad

def L1_function(r, m):
    return r**3 - m * r**2 / (1 - r)**2 - 1

def find_L1(m_prime, initial_guess=0.5, tolerance=1e-12, max_iter=100):

    # thank autodiff!
    L1_derivative = grad(L1_function)

    #first guess : start at halfway point
    r_prime = initial_guess

    for i in range(max_iter):
        # get f and derivative of f 
        f = L1_function(r_prime, m_prime)
        derf = L1_derivative(r_prime, m_prime)

        # Newton's method ( xnew = xold - f(x)/f'(x) )
        r_new = r_prime - f / derf

        # Check convergence
        if jnp.abs(r_new - r_prime) < tolerance:
            return r_new  

        r_prime = r_new

    # If bad
    raise ValueError("Newton's method sucks! it totally failed and you should feel bad!!!")


R_earth_moon = 384000        
R_earth_sun = 149597871  
R_jupiter_sun = 756900000

# Calculate L1 distances km

# Earth-Moon 
m_prime_earth_moon = 0.012303146
r_prime_earth_moon = find_L1(m_prime_earth_moon)
L1_distance_earth_moon = r_prime_earth_moon * R_earth_moon
print(f"L1 distance (Earth-Moon): {L1_distance_earth_moon:.2f} km")

# Earth-Sun 
m_prime_earth_sun = 3.003e-6
r_prime_earth_sun = find_L1(m_prime_earth_sun)
L1_distance_earth_sun = r_prime_earth_sun * R_earth_sun
print(f"L1 distance (Earth-Sun): {L1_distance_earth_sun:.2f} km")

# Jupiter-Sun 
m_prime_jupiter_sun = 0.000954588
r_prime_jupiter_sun = find_L1(m_prime_jupiter_sun)
L1_distance_jupiter_sun = r_prime_jupiter_sun * R_jupiter_sun
print(f"L1 distance (Jupiter-Sun): {L1_distance_jupiter_sun:.2f} km")
