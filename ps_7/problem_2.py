import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

def parabolic_step(func, a, b, c):
    fa, fb, fc = func(a), func(b), func(c)
    denom = (b - a) * (fb - fc) - (b - c) * (fb - fa)
    numer = (b - a)**2 * (fb - fc) - (b - c)**2 * (fb - fa)
    
    if np.abs(denom) < 1.e-15:
        return b  
    return b - 0.5 * numer / denom


def brent(func, astart, bstart, cstart, tol=1.e-8, maxiter=1000):
    golden_ratio = (3. - np.sqrt(5)) / 2

    # init bracket
    a, b, c = astart, bstart, cstart
    niter = 0
    bold = b + 2 * tol  # starts the while loop

    xgrid = np.linspace(0, .5, 1000)
    plt.plot(xgrid, func(xgrid), label="Function")

    while np.abs(bold - b) > tol and niter < maxiter:
        bold = b  

        # parabolic step if  conditions met
        if a < b < c:
            b = parabolic_step(func, a, b, c)
            if b < bold:
                c = bold
            else:
                a = bold
            
            plt.plot([bold, b], func(np.array([bold, b])), color='black')
            plt.plot(b, func(b), 'o', color='black')
        
        # apply golden-section search if conditions not met
        else:
            if (b - a) > (c - b):
                x = b - golden_ratio * (b - a)
                if func(x) < func(b):
                    c, b = b, x
                else:
                    a = x
            else:
                x = b + golden_ratio * (c - b)
                if func(x) < func(b):
                    a, b = b, x
                else:
                    c = x
            #
            plt.plot([b, x], func(np.array([b, x])), color='black')
            plt.plot(x, func(x), 'o', color='black')
        
        niter += 1

    plt.legend()
    return b

def test_function(x):
    return (x - 0.3)**2 * np.exp(x)


a, b, c = 0, 0.5, 1   # create bracket
min_x_brent = brent(test_function, a, b, c)
min_y_brent = test_function(min_x_brent)

min_x_scipy = optimize.brent(test_function)
min_y_scipy = test_function(min_x_scipy)

print("Custom Brent's method implementation:")
print(f"Minimum x: {min_x_brent}")
print(f"Minimum y: {min_y_brent}")

print("\nSciPy's Brent's method implementation:")
print(f"Minimum x: {min_x_scipy}")
print(f"Minimum y: {min_y_scipy}")

plt.plot(min_x_brent, min_y_brent, 'ro', label="Custom Brent's min")
plt.plot(min_x_scipy, min_y_scipy, 'bx', label="SciPy's min")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Finding the Minimum of $f(x) = (x -0.3)^2e^x$")
plt.legend()
plt.show()
