
import numpy as np
import matplotlib.pyplot as plt
import timeit

N_values = list(range(10, 200, 10))  # array containing the sizes of matrices
elapsed_times = []  #place to dump times method 1
elapsed_times_ = [] #place to dump times method 2

#func for loop multiplication
def loop(A,B,C,N):
    for i in range(N):
        for j in range(N):
            for k in range(N):
                C[i, j] += A[i, k] * B[k, j]

#func for np.dot multiplication        
def dot(A,B) :
  C = np.dot(A, B)


#time the loop method for all the N's
print('loopin!')
for N in N_values:
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = np.zeros([N, N], dtype = np.float32)
    t = timeit.timeit(lambda: loop(A,B,C,N), number=3)/3
    elapsed_times.append(t)

#time the dot method for all the N's
print('dottin!')
for N in N_values:
    A = np.random.rand(N, N)
    B = np.random.rand(N, N)
    C = np.zeros([N, N], dtype = np.float32)
    t_ = timeit.timeit(lambda: dot(A,B), number=3)/3
    elapsed_times_.append(t_)


#plot N vs computational time for nested loop method 
plt.plot(N_values, elapsed_times, marker='o')
plt.xlabel('N')
plt.ylabel('Elapsed Time (seconds)')
plt.title('N vs Elapsed Time for Matrix Multiplication')
plt.grid(True)
plt.show()

#plot N vs computational time for np.dot() method
plt.plot(N_values, elapsed_times_, marker='o')
plt.xlabel('N')
plt.ylabel('Elapsed Time (seconds)')
plt.title('N vs Elapsed Time for Numpy Matrix Multiplication')
plt.grid(True)
plt.show()


#plot loglog to show relationship between N and t for each 
plt.loglog(N_values, elapsed_times, marker='o', label='Nested Loops')
#uncommenting this line would put these plots in the same graph and i want them separate
#plt.loglog(N_values, elapsed_times_, marker='x', label='np.dot()')


plt.xlabel('log(N)')
plt.ylabel('log(Elapsed Time)')
plt.title('log(N) vs log(Elapsed Time) for Matrix Multiplication')
plt.grid(True)  
plt.show()

plt.loglog(N_values, elapsed_times_, marker='x', label='np.dot()')


plt.xlabel('log(N)')
plt.ylabel('log(Elapsed Time)')
plt.title('log(N) vs log(Elapsed Time) for Matrix Multiplication')
plt.grid(True)
plt.show()