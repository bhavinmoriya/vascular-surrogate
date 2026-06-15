import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================================
# Configuration
# ============================================================================
class Config:
    nu = 0.01 / np.pi           # Viscosity coefficient
    t_min, t_max = 0.0, 1.0     # Time domain
    x_min, x_max = -1.0, 1.0    # Spatial domain
    hidden_layers = [20, 20, 20, 20]  # 4 hidden layers, 20 neurons each
    input_dim = 2               # (t, x)
    output_dim = 1              # u(t, x)
    lr = 0.001                  # Adam learning rate
    epochs = 5000               # Training epochs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()
print(f"Using device: {config.device}")

# ============================================================================
# PINN Model (Fully-connected MLP with Tanh activation)
# ============================================================================
class PINN(nn.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer: (t, x) -> hidden[0]
        self.layers.append(nn.Linear(config.input_dim, hidden_layers[0]))
        self.layers.append(nn.Tanh())

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.Tanh())

        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], config.output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ============================================================================
# Training Data Generation
# ============================================================================
def generate_training_data():
    np.random.seed(42)

    # Initial condition: 250 points at t=0
    N_ic = 250
    t_ic = np.zeros((N_ic, 1))
    x_ic = np.random.uniform(config.x_min, config.x_max, (N_ic, 1))
    u_ic = -np.sin(np.pi * x_ic)

    # Boundary conditions: 250 points at each boundary
    N_bc = 250
    t_bc = np.random.uniform(config.t_min, config.t_max, (2*N_bc, 1))
    x_bc_left = np.full((N_bc, 1), config.x_min)
    x_bc_right = np.full((N_bc, 1), config.x_max)
    x_bc = np.vstack([x_bc_left, x_bc_right])
    u_bc = np.zeros((2*N_bc, 1))

    # Collocation points: 10,000 for PDE residual
    N_col = 10000
    t_col = np.random.uniform(config.t_min, config.t_max, (N_col, 1))
    x_col = np.random.uniform(config.x_min, config.x_max, (N_col, 1))

    # Convert to tensors
    t_ic, x_ic, u_ic = torch.tensor(t_ic, dtype=torch.float32), \
                       torch.tensor(x_ic, dtype=torch.float32), \
                       torch.tensor(u_ic, dtype=torch.float32)
    t_bc, x_bc, u_bc = torch.tensor(t_bc, dtype=torch.float32), \
                       torch.tensor(x_bc, dtype=torch.float32), \
                       torch.tensor(u_bc, dtype=torch.float32)
    t_col, x_col = torch.tensor(t_col, dtype=torch.float32), \
                   torch.tensor(x_col, dtype=torch.float32)

    return (t_ic, x_ic, u_ic), (t_bc, x_bc, u_bc), (t_col, x_col)

# ============================================================================
# PDE Residual Computation (Burgers' Equation)
# ============================================================================
def compute_pde_residual(model, t_col, x_col):
    """
    Burgers': u_t + u * u_x - nu * u_xx = 0
    Uses automatic differentiation with create_graph=True for 2nd derivatives.
    """
    inputs = torch.cat([t_col, x_col], dim=1)
    inputs.requires_grad = True

    u = model(inputs)

    # First derivatives: u_t, u_x
    u_t = torch.autograd.grad(u, inputs,
                               grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0][:, 0]
    u_x = torch.autograd.grad(u, inputs,
                               grad_outputs=torch.ones_like(u),
                               create_graph=True, retain_graph=True)[0][:, 1]

    # Second derivative: u_xx
    u_xx = torch.autograd.grad(u_x, inputs,
                                grad_outputs=torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0][:, 1]

    residual = u_t + u * u_x - config.nu * u_xx
    return residual, u

# ============================================================================
# Loss Function
# ============================================================================
def compute_loss(model, ic_data, bc_data, col_data):
    t_ic, x_ic, u_ic_true = ic_data
    inputs_ic = torch.cat([t_ic, x_ic], dim=1)
    loss_ic = torch.mean((model(inputs_ic) - u_ic_true) ** 2)

    t_bc, x_bc, u_bc_true = bc_data
    inputs_bc = torch.cat([t_bc, x_bc], dim=1)
    loss_bc = torch.mean((model(inputs_bc) - u_bc_true) ** 2)

    t_col, x_col = col_data
    residual, _ = compute_pde_residual(model, t_col, x_col)
    loss_physics = torch.mean(residual ** 2)

    return loss_ic + loss_bc + loss_physics, loss_ic, loss_bc, loss_physics

# ============================================================================
# Training Loop
# ============================================================================
def train_pinn():
    print(f"Training on {config.device}")

    ic_data, bc_data, col_data = generate_training_data()
    ic_data = tuple(d.to(config.device) for d in ic_data)
    bc_data = tuple(d.to(config.device) for d in bc_data)
    col_data = tuple(d.to(config.device) for d in col_data)

    model = PINN(config.hidden_layers).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    loss_history = []
    loss_ic_hist, loss_bc_hist, loss_phys_hist = [], [], []

    print("Training progress:")
    for epoch in range(config.epochs):
        loss_total, loss_ic, loss_bc, loss_physics = compute_loss(
            model, ic_data, bc_data, col_data)

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        loss_history.append(loss_total.item())
        loss_ic_hist.append(loss_ic.item())
        loss_bc_hist.append(loss_bc.item())
        loss_phys_hist.append(loss_physics.item())

        if epoch % 500 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch:5d} | Total: {loss_total.item():.6e} | "
                  f"IC: {loss_ic.item():.6e} | BC: {loss_bc.item():.6e} | "
                  f"Physics: {loss_physics.item():.6e}")

    return model, (loss_history, loss_ic_hist, loss_bc_hist, loss_phys_hist)

# ============================================================================
# Visualization
# ============================================================================
def visualize_results(model, loss_history):
    loss_total = loss_history[0]

    t_pred = np.linspace(config.t_min, config.t_max, 100)
    x_pred = np.linspace(config.x_min, config.x_max, 100)

    t_grid = np.tile(t_pred, (len(x_pred), 1))
    x_grid = np.tile(x_pred, (len(t_pred), 1)).T

    inputs = torch.tensor(np.vstack([t_grid.flatten(), x_grid.flatten()]),
                          dtype=torch.float32).to(config.device).T
    u_pred = model(inputs).cpu().detach().numpy().flatten()
    u_pred = u_pred.reshape(len(t_pred), len(x_pred))

    # Contour plot
    plt.figure(figsize=(8, 6))
    plt.contourf(t_pred, x_pred, u_pred, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('PINN Prediction for Burgers\' Equation')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.savefig('output/burgers_solution.png', dpi=150)
    plt.close()

    # Loss history
    plt.figure(figsize=(10, 4))
    plt.plot(loss_total, label='Total')
    plt.plot(loss_history[1], label='IC', alpha=0.7)
    plt.plot(loss_history[2], label='BC', alpha=0.7)
    plt.plot(loss_history[3], label='Physics', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')
    plt.savefig('output/loss_history.png', dpi=150)
    plt.close()

    print("Saved: output/burgers_solution.png, output/loss_history.png")

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)

    print("\nTraining PINN...")
    model, loss_history = train_pinn()

    torch.save(model.state_dict(), 'output/pinn_burgers.pth')
    print(f"Model saved to output/pinn_burgers.pth")

    print("\nVisualizing...")
    visualize_results(model, loss_history)

    print("\nDone!")
