###Problem 2 on PS2

import numpy as np

eps32 = np.float32(2.0e-23)
eps64 = np.float64(2.0e-53)

one_32 = np.float32(1.0)
one_64 = np.float64(1.0)

while np.float32(one_32 + eps32) == np.float32(1.0):
  eps32 *= np.float32(1.001)

while np.float64(one_64 + eps64) == np.float64(1.0):
  eps64 *= np.float64(1.001)


min_32 = np.finfo(np.float32).tiny
min_64 = np.finfo(np.float64).tiny
max_32 = np.finfo(np.float32).max
max_64 = np.finfo(np.float64).max


print(f"Minimum positive number in 32-bit precision: {min_32}")
print(f"Minimum positive number in 64 bit precision: {min_64}")

print(f"Maximum positive number in 32-bit precision: {max_32}")
print(f"Maximum positive number in 64-bit precision: {max_64}")

print('The smallest number greater than one in 32 bit precision: ',one_32+ eps32)
print('The smallest value that can be added to one in 32 bit precision: ', eps32)
print('The smallest number greater than one in 64 bit precision: ',one_64 + eps64)
print('The smallest value that can be added to one in 64 bit precision: ', eps64)