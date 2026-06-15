**Reduced-Order Models (ROMs)** are simplified mathematical models that approximate a much larger and more computationally expensive system while preserving its essential behavior.

The main idea is:

> Instead of solving a system with millions of variables, solve a much smaller system with perhaps tens or hundreds of variables.

### Why ROMs are useful

Suppose you're simulating:

* Fluid dynamics around an aircraft
* Weather prediction
* Structural deformation of a bridge
* Electromagnetic fields
* Battery physics

A high-fidelity simulation may take:

* Hours or days
* Large clusters of CPUs/GPUs

A ROM can often produce a reasonably accurate answer in:

* Milliseconds to seconds

This makes ROMs valuable for:

* Real-time control
* Optimization
* Digital twins
* Parameter sweeps
* Uncertainty quantification

---

## Intuition

Imagine recording a person's movement.

A full model tracks:

* Position of every muscle fiber
* Every joint
* Every tendon

A reduced model might capture:

* Walking speed
* Stride length
* Direction

Much less information, but still enough to predict where the person will be.

---

## Mathematical View

A large dynamical system may be:

[
\dot{x} = f(x,u)
]

where:

* (x \in \mathbb{R}^N)
* (N) may be millions

ROM assumes the solution approximately lies in a low-dimensional subspace:

[
x \approx V a
]

where:

* (V) = basis matrix
* (a \in \mathbb{R}^r)
* (r \ll N)

For example:

* Original dimension: (N = 1,000,000)
* Reduced dimension: (r = 50)

The reduced system becomes:

[
\dot{a}=V^T f(Va,u)
]

which is dramatically cheaper to solve.

---

## Common ROM Techniques

### 1. Proper Orthogonal Decomposition (POD)

Most popular ROM method.

Steps:

1. Run expensive simulations
2. Collect solution snapshots
3. Perform SVD

[
X = U\Sigma V^T
]

4. Keep only dominant singular vectors

These vectors become the reduced basis.

Statistically, POD is very similar to:

* Principal Component Analysis (PCA)

The retained energy is:

[
\frac{\sum_{i=1}^{r}\sigma_i^2}
{\sum_{i=1}^{N}\sigma_i^2}
]

A common choice is to retain 99% of the energy.

---

### 2. Balanced Truncation

Popular in control theory.

Keeps states that are:

* Easily controllable
* Easily observable

Removes states with little influence on system behavior.

Used extensively in aerospace and robotics.

---

### 3. Krylov Subspace Methods

Examples:

* Arnoldi
* Lanczos

Common in:

* Circuit simulation
* Electromagnetics

---

### 4. Dynamic Mode Decomposition (DMD)

Extracts dominant dynamic patterns directly from data.

Widely used in:

* Fluid mechanics
* Video analysis
* Time-series forecasting

---

### 5. Autoencoder-Based ROMs

Modern machine learning approach.

Instead of linear reduction:

[
x \approx Decoder(z)
]

where:

* (z) is a low-dimensional latent vector

The encoder learns nonlinear manifolds that classical POD cannot capture.

Used in physics-informed ML and scientific machine learning.

---

## Example: Fluid Flow

A computational fluid dynamics (CFD) simulation may contain:

* 5 million mesh cells

Each timestep solves millions of equations.

Using POD:

* Collect velocity fields
* Extract 20 dominant modes
* Simulate only those modes

Result:

* Speed-up of 100–10000×
* Often with only a few percent error

---

## Relationship to Machine Learning

ROMs and ML are becoming increasingly connected.

| Classical ROM       | Machine Learning Analogy |
| ------------------- | ------------------------ |
| POD                 | PCA                      |
| Reduced basis       | Feature extraction       |
| State compression   | Latent representation    |
| Galerkin projection | Learned dynamics         |
| Autoencoder ROM     | Deep latent models       |

A useful perspective is:

> ROMs are physics-driven dimensionality reduction.

while

> ML often performs data-driven dimensionality reduction.

---

## If you're interviewing for ML/AI roles

A concise answer:

> "A Reduced-Order Model (ROM) is a low-dimensional approximation of a high-fidelity physical or dynamical system. ROMs use techniques such as Proper Orthogonal Decomposition (POD), balanced truncation, or autoencoders to project the original high-dimensional state into a much smaller space, enabling faster simulation, optimization, and real-time control while retaining most of the system's important dynamics."

This definition is commonly expected in scientific ML, physics-informed AI, digital twins, computational engineering, and simulation-heavy AI roles.
