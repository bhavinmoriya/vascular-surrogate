**PINN (Physics-Informed Neural Network)** is a neural network that learns not only from data but also from the **physical laws governing the system**.

A traditional ANN asks:

> "What input-output pairs do I observe?"

A PINN asks:

> "What input-output pairs do I observe, and what physical laws must the solution satisfy?"

---

## The Core Idea

Suppose you want to model temperature in a metal rod.

A standard ANN learns:

[
(x,t) \rightarrow T(x,t)
]

using measured temperatures.

A PINN learns the same mapping but additionally enforces the **heat equation**:

$$
[
\frac{\partial T}{\partial t} =
\alpha
\frac{\partial^2 T}{\partial x^2}
]
$$

during training.

The network is penalized whenever its predictions violate this equation.

---

## How PINNs Work

A standard ANN loss might be:

[
Loss = Loss_{data}
]

For PINNs:

[
Loss =
Loss_{data}
+
\lambda Loss_{physics}
]

where

[
Loss_{physics}
==============

\left|
\frac{\partial T}{\partial t}
-----------------------------

\alpha
\frac{\partial^2 T}{\partial x^2}
\right|^2
]

Automatic differentiation computes these derivatives directly through the network.

---

### Visualization

The governing equation is central to the method:

\frac{\partial T}{\partial t}=\alpha\frac{\partial^2T}{\partial x^2}

The network is trained so that its predictions satisfy this relationship throughout the domain.

---

## Why is PINN Useful?

### 1. Requires Less Data

ANNs are data-hungry.

Example:

* ANN may require 100,000 simulation samples.
* PINN may work with hundreds or thousands because the physics provides additional information.

Think of it as:

* ANN learns from examples.
* PINN learns from examples **plus a textbook of physical laws**.

---

### 2. Generalizes Better

ANN can produce physically impossible results.

Example:

* Negative density
* Violation of conservation laws
* Nonphysical temperatures

PINNs are encouraged to obey the underlying equations, making such errors less likely.

---

### 3. Can Solve PDEs Without Simulation Data

Traditional numerical methods:

* Finite Element Method (FEM)
* Finite Difference Method (FDM)
* Finite Volume Method (FVM)

need discretization and mesh generation.

PINNs can solve many PDE problems by learning a function that satisfies:

* Differential equation
* Boundary conditions
* Initial conditions

even when no labeled solution data exists.

---

### 4. Works with Sparse Measurements

Imagine you have:

* Temperature sensors at only 10 locations

but want the full temperature field.

PINNs can infer the missing regions by leveraging the governing physics.

This is called an **inverse problem**.

---

### 5. Parameter Discovery

Suppose the equation is

[
\frac{\partial u}{\partial t}
=============================

k
\frac{\partial^2u}{\partial x^2}
]

but you don't know (k).

PINNs can learn:

* the solution (u)
* the unknown parameter (k)

simultaneously.

This is valuable in science and engineering.

---

## ANN vs PINN

| Feature                      | ANN            | PINN       |
| ---------------------------- | -------------- | ---------- |
| Uses data                    | ✓              | ✓          |
| Uses physics equations       | ✗              | ✓          |
| Data requirement             | High           | Lower      |
| Physical consistency         | Not guaranteed | Encouraged |
| PDE solving                  | Poor           | Strong     |
| Unknown parameter estimation | Difficult      | Natural    |
| Training speed               | Faster         | Slower     |

---

## Why PINNs Are Not Always Better

PINNs sound amazing, but they have limitations.

### Training is harder

The loss contains:

* Data loss
* PDE residual loss
* Boundary condition loss

Balancing them can be difficult.

---

### Often slower

A standard ANN only computes outputs.

PINNs repeatedly compute derivatives via automatic differentiation, which is computationally expensive.

---

### Struggle with Complex PDEs

For highly turbulent fluid flows or very stiff systems, classical solvers can still outperform PINNs.

Many current research efforts focus on improving this.

---

## Where PINNs Are Used

* Fluid dynamics
* Weather modeling
* Aerodynamics
* Battery modeling
* Material science
* Electromagnetics
* Structural mechanics
* Digital twins
* Scientific machine learning

Companies building simulation-based AI systems often explore PINNs because generating simulation data can be extremely expensive.

---

## Intuition from Feynman's Perspective

Imagine teaching a student projectile motion.

### ANN

You show thousands of examples:

```
velocity → trajectory
velocity → trajectory
velocity → trajectory
```

The student memorizes patterns.

### PINN

You also teach:

[
F = ma
]

and the laws of motion.

Now the student can reason about trajectories even in situations they have never seen before.

That is essentially what a Physics-Informed Neural Network does: it combines pattern learning with known physical laws.
