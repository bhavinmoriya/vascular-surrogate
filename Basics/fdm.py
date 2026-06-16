import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10  # Number of grid points
L = 1.0  # Length of the domain
h = L / (N - 1)  # Grid spacing
x = np.linspace(0, L, N)  # Grid points

# Source term (f(x) = -2)
f = -2 * np.ones(N)

# Initialize the coefficient matrix A and right-hand side vector b
A = np.zeros((N, N))
b = np.zeros(N)

# Apply FDM to approximate the second derivative
for i in range(1, N - 1):
    A[i, i - 1] = 1
    A[i, i] = -2
    A[i, i + 1] = 1
    b[i] = h**2 * f[i]

# Apply boundary conditions: u(0) = 0, u(1) = 0
A[0, 0] = 1
A[N - 1, N - 1] = 1
b[0] = 0
b[N - 1] = 0

# Solve the linear system
u = np.linalg.solve(A, b)

# Exact solution for comparison
u_exact = x * (1 - x)

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(x, u, 'o-', label='FDM Solution')
plt.plot(x, u_exact, '--', label='Exact Solution')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('1D Poisson Equation: FDM vs Exact Solution')
plt.legend()
plt.grid(True)
plt.show()

# Print the solution
print("Grid points (x):", x)
print("FDM solution (u):", u)
print("Exact solution:", u_exact)
