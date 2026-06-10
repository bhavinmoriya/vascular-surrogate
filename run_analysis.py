"""
run_analysis.py
---------------
End-to-end pipeline:
  1. Generate synthetic CFD dataset (2000 patient geometries)
  2. Train GP, NN, PCE surrogates on 80% split
  3. Evaluate on 20% holdout: R², RMSE, speedup
  4. Plot parity plots, uncertainty, and Sobol sensitivity indices
  5. Save all figures to ./figures/

Run:
    python run_analysis.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cfd_simulator import generate_dataset, run_cfd, AneurysmGeometry
from surrogate_models import GPSurrogate, NNSurrogate, PCESurrogate

import time, warnings
warnings.filterwarnings("ignore")

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

SEED = 42
rng  = np.random.default_rng(SEED)

COLORS = {
    "GP":  "#2E86AB",
    "NN":  "#E84855",
    "PCE": "#3BB273",
    "CFD": "#F4A261",
}

# ============================================================
print("=" * 60)
print("VASCULAR SURROGATE MODEL BENCHMARK")
print("=" * 60)

# ---- 1. Generate dataset -----------------------------------
print("\n[1/5] Generating CFD dataset (n=2000)...")
t0 = time.perf_counter()
X, y, feat_names, tgt_names = generate_dataset(n=2000, seed=SEED)
cfd_time_per_sample = (time.perf_counter() - t0) / 2000  # seconds per sample (analytic proxy)
print(f"      Dataset shape: X={X.shape}, y={y.shape}")
print(f"      Simulated CFD oracle time (analytic proxy): {cfd_time_per_sample*1000:.3f} ms/sample")

# ---- 2. Split -----------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f"\n[2/5] Train: {X_tr.shape[0]}  Test: {X_te.shape[0]}")

# Use smaller GP training set for speed (GP scales as O(n³))
X_tr_gp, _, y_tr_gp, _ = train_test_split(X_tr, y_tr, train_size=300, random_state=SEED)

# ---- 3. Train models ----------------------------------------
print("\n[3/5] Training surrogates...")

models = {}
train_times = {}

for name, model, Xtr, ytr in [
    ("GP",  GPSurrogate(n_restarts=3),         X_tr_gp, y_tr_gp),
    ("NN",  NNSurrogate(max_iter=500, seed=SEED), X_tr,    y_tr),
    ("PCE", PCESurrogate(degree=2),              X_tr,    y_tr),
]:
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    dt = time.perf_counter() - t0
    train_times[name] = dt
    models[name] = model
    print(f"  {name:4s}  trained in {dt:.2f}s  (n_train={Xtr.shape[0]})")

# ---- 4. Evaluate --------------------------------------------
print("\n[4/5] Evaluating on holdout set...")

scores = {}
infer_times = {}
for name, model in models.items():
    t0 = time.perf_counter()
    sc = model.score(X_te, y_te)
    dt = (time.perf_counter() - t0) / len(X_te)
    scores[name] = sc
    infer_times[name] = dt
    r2_mean = sc["r2"].mean()
    print(f"  {name:4s}  mean R²={r2_mean:.4f}  inference {dt*1e6:.1f} µs/sample")

# Speedup: ratio of CFD oracle time to surrogate inference time
print("\n  --- Speedup vs CFD oracle ---")
for name in models:
    su = cfd_time_per_sample / infer_times[name]
    print(f"  {name:4s}  {su:.0f}× faster")

# ---- 5. Figures ---------------------------------------------
print("\n[5/5] Generating figures...")

TARGET_LABELS = {
    "max_wss":           "Max WSS [Pa]",
    "mean_wss":          "Mean WSS [Pa]",
    "min_wss":           "Min WSS [Pa]",
    "osi":               "OSI [-]",
    "pressure_drop":     "Pressure Drop [Pa]",
    "vortex_strength":   "Vortex Strength [-]",
    "rupture_risk_score":"Rupture Risk Score [-]",
}

# --- Fig 1: R² comparison bar chart -------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Surrogate Model Benchmark — Intracranial Aneurysm Haemodynamics",
             fontsize=13, fontweight="bold", y=1.01)

ax = axes[0]
x = np.arange(len(tgt_names))
w = 0.25
for i, (name, color) in enumerate(COLORS.items()):
    if name == "CFD": continue
    r2 = scores[name]["r2"]
    ax.bar(x + i*w, r2, w, label=name, color=color, alpha=0.88, edgecolor='white')
ax.set_xticks(x + w)
ax.set_xticklabels([TARGET_LABELS[t].split(" [")[0] for t in tgt_names], rotation=30, ha='right', fontsize=8)
ax.set_ylabel("R²", fontsize=11)
ax.set_title("Test-set R² per output", fontsize=11)
ax.axhline(0.95, ls='--', color='gray', lw=0.8, label='R²=0.95 reference')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# --- Fig 1b: RMSE comparison --------------------------------
ax2 = axes[1]
for i, (name, color) in enumerate(COLORS.items()):
    if name == "CFD": continue
    rmse = scores[name]["rmse"]
    rmse_norm = rmse / (y_te.max(0) - y_te.min(0) + 1e-12)  # normalised RMSE
    ax2.bar(x + i*w, rmse_norm, w, label=name, color=color, alpha=0.88, edgecolor='white')
ax2.set_xticks(x + w)
ax2.set_xticklabels([TARGET_LABELS[t].split(" [")[0] for t in tgt_names], rotation=30, ha='right', fontsize=8)
ax2.set_ylabel("Normalised RMSE", fontsize=11)
ax2.set_title("Normalised RMSE per output", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "fig1_accuracy_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig1_accuracy_comparison.png")

# --- Fig 2: Parity plots for rupture risk (most clinically relevant) --------
target_idx = tgt_names.index("rupture_risk_score")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Parity Plots — Rupture Risk Score (Holdout Set)", fontsize=12, fontweight='bold')

for ax, (name, color) in zip(axes, [(k, v) for k, v in COLORS.items() if k != "CFD"]):
    y_pred = models[name].predict(X_te)[:, target_idx]
    y_true = y_te[:, target_idx]
    r2 = scores[name]["r2"][target_idx]
    ax.scatter(y_true, y_pred, s=8, alpha=0.5, color=color, rasterized=True)
    mn, mx = y_true.min(), y_true.max()
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1.2, label='Perfect fit')
    ax.set_xlabel("CFD Oracle (true)", fontsize=10)
    ax.set_ylabel("Surrogate Prediction", fontsize=10)
    ax.set_title(f"{name}  (R²={r2:.4f})", fontsize=11, color=color, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "fig2_parity_rupture_risk.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig2_parity_rupture_risk.png")

# --- Fig 3: GP uncertainty (calibration) ------------------------------------
gp_mean, gp_std = models["GP"].predict_with_uncertainty(X_te)
tgt = tgt_names.index("max_wss")

errors = np.abs(gp_mean[:, tgt] - y_te[:, tgt])
sigma  = gp_std[:, tgt]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Gaussian Process Uncertainty Quantification — Max WSS", fontsize=12, fontweight='bold')

# Calibration: |error| vs predicted std
ax = axes[0]
ax.scatter(sigma, errors, s=8, alpha=0.4, color=COLORS["GP"])
xlim = ax.get_xlim()
ax.plot([0, xlim[1]], [0, xlim[1]], 'k--', lw=1.2, label='|error|=σ')
ax.plot([0, xlim[1]], [0, 2*xlim[1]], 'r--', lw=0.8, alpha=0.5, label='|error|=2σ')
ax.set_xlabel("Predicted σ [Pa]", fontsize=10)
ax.set_ylabel("|Prediction Error| [Pa]", fontsize=10)
ax.set_title("Error vs Predicted Uncertainty", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Sorted prediction intervals
idx = np.argsort(y_te[:, tgt])[:200]
x_plot = np.arange(200)
ax2 = axes[1]
ax2.fill_between(x_plot,
                 gp_mean[idx, tgt] - 2*gp_std[idx, tgt],
                 gp_mean[idx, tgt] + 2*gp_std[idx, tgt],
                 alpha=0.25, color=COLORS["GP"], label='±2σ')
ax2.plot(x_plot, y_te[idx, tgt], 'o', ms=3, color='black', label='CFD truth', alpha=0.6)
ax2.plot(x_plot, gp_mean[idx, tgt], '-', color=COLORS["GP"], lw=1.5, label='GP mean')
ax2.set_xlabel("Sorted test samples", fontsize=10)
ax2.set_ylabel("Max WSS [Pa]", fontsize=10)
ax2.set_title("Prediction Intervals (sorted)", fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "fig3_gp_uncertainty.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig3_gp_uncertainty.png")

# --- Fig 4: PCE Sobol Sensitivity Indices -----------------------------------
sobol = models["PCE"].sobol_first_order()   # [n_inputs, n_outputs]

fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(sobol.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=sobol.max())
ax.set_xticks(range(len(feat_names)))
ax.set_xticklabels(feat_names, rotation=40, ha='right', fontsize=9)
ax.set_yticks(range(len(tgt_names)))
ax.set_yticklabels([TARGET_LABELS[t] for t in tgt_names], fontsize=9)
plt.colorbar(im, ax=ax, label='First-order Sobol index S_i')
ax.set_title("Global Sensitivity Analysis — PCE First-Order Sobol Indices\n"
             "Rows: QoI outputs   Columns: input parameters", fontsize=11, fontweight='bold')
for i in range(sobol.shape[0]):
    for j in range(sobol.shape[1]):
        ax.text(i, j, f"{sobol[i,j]:.2f}", ha='center', va='center', fontsize=7,
                color='white' if sobol[i,j] > 0.3 else 'black')
plt.tight_layout()
plt.savefig(OUT / "fig4_sobol_sensitivity.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig4_sobol_sensitivity.png")

# --- Fig 5: Inference speedup bar -------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
names_plot = list(models.keys())
colors_plot = [COLORS[n] for n in names_plot]
speedups = [cfd_time_per_sample / infer_times[n] for n in names_plot]
bars = ax.bar(names_plot, speedups, color=colors_plot, edgecolor='white', width=0.5)
for bar, su in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{su:.0f}×", ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel("Speedup over CFD oracle", fontsize=11)
ax.set_title("Surrogate Inference Speedup\n(relative to CFD wall-time proxy)", fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(speedups) * 1.25)
plt.tight_layout()
plt.savefig(OUT / "fig5_speedup.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig5_speedup.png")

# ---- Print final summary table ------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
header = f"{'Model':5s}  {'Mean R²':>8s}  {'Mean RMSE':>10s}  {'Infer [µs]':>12s}"
print(header)
print("-" * 50)
for name in models:
    r2m   = scores[name]["r2"].mean()
    rmsem = scores[name]["rmse"].mean()
    it    = infer_times[name] * 1e6
    print(f"{name:5s}  {r2m:8.4f}  {rmsem:10.4f}  {it:12.2f}")
print("=" * 60)
print("\nDone. All figures saved to ./figures/")
