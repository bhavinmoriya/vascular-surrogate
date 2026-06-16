Here’s a **simple Python implementation** of the **Finite Difference Method (FDM)** to solve the **1D steady-state heat equation** (Poisson equation) with Dirichlet boundary conditions. This is a classic example to introduce FDM for solving PDEs.

---

### **Problem Statement**
Solve the 1D Poisson equation:
\[
\frac{d^2 u}{dx^2} = f(x), \quad \text{for} \quad x \in [0, 1]
\]
with boundary conditions:
\[
u(0) = 0, \quad u(1) = 0
\]
and source term:
\[
f(x) = -2 \quad (\text{constant})
\]
The exact solution for this problem is:
\[
u(x) = x(1 - x)
\]

---

### **FDM Approach**
1. **Discretize the domain**: Divide \([0, 1]\) into \(N\) equally spaced points with spacing \(h = \frac{1}{N-1}\).
2. **Approximate the second derivative** using the central difference formula:
   \[
   \frac{d^2 u}{dx^2} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{h^2}
   \]
3. **Set up the linear system**: For each interior point \(i\), the equation becomes:
   \[
   \frac{u_{i+1} - 2u_i + u_{i-1}}{h^2} = f(x_i)
   \]
   Rearrange to form a system of linear equations:
   \[
   u_{i-1} - 2u_i + u_{i+1} = h^2 f(x_i)
   \]
4. **Apply boundary conditions**: \(u_0 = 0\) and \(u_N = 0\).
5. **Solve the system** for the unknowns \(u_1, u_2, \dots, u_{N-1}\).

---

### **Python Code**
```python
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
```

---

### **Explanation of the Code**
1. **Grid Setup**:
   - `N` is the number of grid points.
   - `h` is the spacing between points.
   - `x` is the array of grid points from 0 to 1.

2. **Source Term**:
   - `f` is the right-hand side of the Poisson equation (here, a constant \(-2\)).

3. **Matrix Assembly**:
   - The matrix `A` represents the discretized second derivative operator.
   - The vector `b` contains the source term multiplied by \(h^2\).

4. **Boundary Conditions**:
   - The first and last rows of `A` are set to enforce \(u(0) = 0\) and \(u(1) = 0\).

5. **Solving the System**:
   - `np.linalg.solve` is used to solve the linear system \(A \mathbf{u} = \mathbf{b}\).

6. **Comparison with Exact Solution**:
   - The exact solution \(u(x) = x(1 - x)\) is plotted alongside the FDM solution for validation.

---

### **Output**
- The code will plot the FDM solution (circles and lines) and the exact solution (dashed line).
- You’ll see that the FDM solution matches the exact solution closely, especially for larger \(N\) (e.g., \(N = 20\) or \(N = 50\)).

---

### **Key Observations**
- For \(N = 10\), the solution is already quite accurate.
- Increasing \(N\) (e.g., to 50 or 100) will improve accuracy but increase computational cost.
- FDM is **simple to implement** for regular grids and linear PDEs.

---
Would you like to extend this to a **2D problem** (e.g., Laplace equation) or explore how to handle **Neumann boundary conditions**?
