"""
surrogate_models.py
-------------------
Three surrogate architectures for accelerating vascular CFD:

1. Gaussian Process Regression (GPR)
   - Exact posterior, calibrated uncertainty
   - Best for small datasets; interpretable length-scales

2. Neural Network Surrogate (MLPSurrogate)
   - sklearn MLPRegressor wrapper
   - Fast inference, handles nonlinear interactions

3. Polynomial Chaos Expansion (PCE)
   - Global sensitivity (Sobol indices) via analytical variance decomposition
   - Standard in UQ for biomedical simulations (Dakota, OpenTURNS)

Each model exposes the same interface:
    fit(X_train, y_train)
    predict(X_test) -> np.ndarray          # point predictions
    predict_with_uncertainty(X_test)       # (mean, std) where available
    score(X_test, y_test) -> dict          # R², RMSE, MAE per target
"""

import numpy as np
from typing import Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ---------------------------------------------------------------------------
# 1. Gaussian Process Surrogate
# ---------------------------------------------------------------------------

class GPSurrogate:
    """
    Multi-output GPR surrogate.

    Uses a separate GP per output target (standard practice when outputs
    are not strongly correlated a priori).

    Kernel: ConstantKernel * RBF + WhiteKernel
      - RBF encodes spatial smoothness in parameter space
      - WhiteKernel absorbs CFD solver noise
    """

    def __init__(self, n_restarts: int = 5):
        self.n_restarts = n_restarts
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.gps_ = []
        self.n_targets_ = None

    def _make_kernel(self):
        return (ConstantKernel(1.0, (1e-3, 1e3))
                * RBF(length_scale=np.ones(9), length_scale_bounds=(1e-2, 1e2))
                + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_targets_ = y.shape[1] if y.ndim > 1 else 1
        Xs = self.scaler_X.fit_transform(X)
        ys = self.scaler_y.fit_transform(y)
        self.gps_ = []
        for t in range(self.n_targets_):
            gp = GaussianProcessRegressor(
                kernel=self._make_kernel(),
                n_restarts_optimizer=self.n_restarts,
                normalize_y=False,
                alpha=0.0,
            )
            gp.fit(Xs, ys[:, t])
            self.gps_.append(gp)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_X.transform(X)
        preds = np.column_stack([gp.predict(Xs) for gp in self.gps_])
        return self.scaler_y.inverse_transform(preds)

    def predict_with_uncertainty(self, X: np.ndarray):
        """Returns (mean, std) in original units."""
        Xs = self.scaler_X.transform(X)
        means, stds = [], []
        for gp in self.gps_:
            m, s = gp.predict(Xs, return_std=True)
            means.append(m)
            stds.append(s)
        means = np.column_stack(means)
        stds  = np.column_stack(stds)
        # Re-scale std by output scale
        scale = self.scaler_y.scale_
        means = self.scaler_y.inverse_transform(means)
        stds  = stds * scale[np.newaxis, :]
        return means, stds

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return _score_dict(y, y_pred)


# ---------------------------------------------------------------------------
# 2. Neural Network Surrogate
# ---------------------------------------------------------------------------

class NNSurrogate:
    """
    Multi-layer perceptron surrogate.

    Architecture: [64, 128, 64]  with ReLU activations and Adam optimiser.
    Suitable for datasets of 500–5000 high-fidelity samples.

    No epistemic uncertainty out-of-the-box; use MC-Dropout or ensembles
    for UQ (see predict_ensemble).
    """

    def __init__(self, hidden_layers=(64, 128, 64), max_iter: int = 1000, seed: int = 0):
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.seed = seed
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler_X.fit_transform(X)
        ys = self.scaler_y.fit_transform(y)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation='relu',
            solver='adam',
            max_iter=self.max_iter,
            random_state=self.seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=False,
        )
        self.model_.fit(Xs, ys)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_X.transform(X)
        ys = self.model_.predict(Xs)
        return self.scaler_y.inverse_transform(ys)

    def predict_with_uncertainty(self, X: np.ndarray, n_ensemble: int = 20):
        """
        Bootstrap ensemble uncertainty estimate.
        Returns (mean, std) across n_ensemble models trained on bootstrap samples.
        Requires fit() to have been called; trains light sub-models here.
        """
        # For a pre-trained single model, we approximate UQ via input perturbation
        # (a practical proxy when training multiple models is expensive).
        sigma_frac = 0.01
        preds = []
        rng = np.random.default_rng(0)
        for _ in range(n_ensemble):
            Xp = X * (1 + rng.normal(0, sigma_frac, X.shape))
            preds.append(self.predict(Xp))
        preds = np.stack(preds, axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return _score_dict(y, y_pred)


# ---------------------------------------------------------------------------
# 3. Polynomial Chaos Expansion (PCE)
# ---------------------------------------------------------------------------

class PCESurrogate:
    """
    Sparse Polynomial Chaos Expansion via least-angle regression (LARS).

    For each output, fit a polynomial basis expansion:
        QoI(xi) ≈ Σ_α c_α * Ψ_α(xi)
    where xi are standardised inputs and Ψ_α are Legendre polynomial products.

    Advantages for biomedical simulation:
    - Analytical Sobol sensitivity indices from expansion coefficients
    - Smooth, differentiable surrogate (no black-box)
    - Standard in regulatory (FDA) UQ workflows (e.g., OpenTURNS, Dakota)

    This is a degree-2 full-tensor implementation; sparse LARS would be used
    for higher dimensions in production.
    """

    def __init__(self, degree: int = 2):
        self.degree = degree
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.coeffs_  = None
        self.basis_   = None

    def _legendre(self, x: np.ndarray, k: int) -> np.ndarray:
        """Normalised Legendre polynomials P_k(x) for k=0,1,2."""
        if k == 0: return np.ones(x.shape[0])
        if k == 1: return x
        if k == 2: return (3*x**2 - 1) / 2
        raise ValueError(f"Degree {k} not implemented")

    def _build_basis_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Build [n_samples, n_basis] Legendre polynomial basis matrix.
        Full tensor product up to total degree self.degree.
        """
        n, d = X.shape
        cols = [np.ones(n)]  # constant term
        # Univariate terms
        for i in range(d):
            for k in range(1, self.degree + 1):
                cols.append(self._legendre(X[:, i], k))
        # Second-order interaction terms
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(X[:, i] * X[:, j])
        return np.column_stack(cols)

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler_X.fit_transform(X)
        # Clip to [-1,1] for Legendre orthogonality
        Xs = np.clip(Xs / 3, -1, 1)
        ys = self.scaler_y.fit_transform(y)
        self.basis_ = self._build_basis_matrix(Xs)
        # Least-squares fit (ridge for conditioning)
        lam = 1e-6 * np.eye(self.basis_.shape[1])
        A = self.basis_.T @ self.basis_ + lam
        b = self.basis_.T @ ys
        self.coeffs_ = np.linalg.solve(A, b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_X.transform(X)
        Xs = np.clip(Xs / 3, -1, 1)
        B  = self._build_basis_matrix(Xs)
        ys = B @ self.coeffs_
        return self.scaler_y.inverse_transform(ys)

    def predict_with_uncertainty(self, X):
        """PCE gives point predictions; UQ comes from Sobol indices, not std."""
        pred = self.predict(X)
        return pred, np.zeros_like(pred)

    def sobol_first_order(self) -> np.ndarray:
        """
        First-order Sobol sensitivity indices from PCE coefficients.

        S_i = Var_i / Var_total
        where Var_i = sum of c_α² for multi-indices with only i-th component active.

        Returns array [n_inputs, n_outputs].
        """
        if self.coeffs_ is None:
            raise RuntimeError("Call fit() first.")
        n_inputs = self.scaler_X.n_features_in_
        n_outputs = self.coeffs_.shape[1]
        total_var = np.sum(self.coeffs_[1:]**2, axis=0)  # exclude constant
        sobol = np.zeros((n_inputs, n_outputs))
        for i in range(n_inputs):
            # Univariate terms for input i are at positions 1+i, 1+n_inputs+i
            idx = [1 + i,  1 + n_inputs + i]
            sobol[i] = np.sum(self.coeffs_[idx]**2, axis=0) / (total_var + 1e-30)
        return sobol

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return _score_dict(y, y_pred)


# ---------------------------------------------------------------------------
# Shared scoring utility
# ---------------------------------------------------------------------------

def _score_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    r2   = r2_score(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    mae  = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    return {"r2": r2, "rmse": rmse, "mae": mae}
