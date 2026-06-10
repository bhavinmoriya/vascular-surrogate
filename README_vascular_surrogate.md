# Vascular Surrogate Models — Accelerating Intracranial Aneurysm CFD

**Surrogate modelling for haemodynamic simulation of intracranial aneurysms using Gaussian Processes, Neural Networks, and Polynomial Chaos Expansions.**

This project builds and benchmarks data-driven surrogate models that replace computationally expensive CFD and FEM solvers for patient-specific vascular simulation — the core technical challenge in regulatory in-silico trials (e.g., STRIVE, FDA 2023 guidance on computational modelling).

---

## Motivation

Full-fidelity CFD of a single aneurysm geometry takes **hours on an HPC cluster**. In-silico clinical trials require evaluating thousands of patient-specific scenarios. Surrogate models compress this wall-time by orders of magnitude while preserving accuracy — enabling:

- Rapid preoperative treatment planning
- Uncertainty quantification (UQ) across patient populations  
- Device optimisation without repeated high-fidelity runs
- Regulatory submissions aligned with FDA/ISO computational modelling guidelines

---

## Physics Basis

The CFD oracle (`cfd_simulator.py`) implements an analytically grounded approximation of haemodynamic quantities of interest (QoIs) for saccular intracranial aneurysms:

| QoI | Physical model |
|-----|---------------|
| Max / Mean / Min WSS | Poiseuille baseline × geometric amplification (Shojima 2004) |
| Oscillatory Shear Index (OSI) | Womersley-scaled pulsatility model |
| Pressure drop | Bernoulli + viscous (Hagen–Poiseuille) |
| Vortex strength | Reynolds × aspect ratio proxy |
| Rupture risk score | Composite index (Xiang et al. 2011 weights) |

**Input parameters** (sampled from clinical literature distributions):

| Parameter | Range | Source |
|-----------|-------|--------|
| Neck diameter | 2–8 mm | ISUIA cohort |
| Sac diameter | 3–15 mm | Morphological studies |
| Parent artery diameter | 2–5 mm | ICA/MCA anatomy |
| Inflow velocity | 0.2–0.6 m/s | PC-MRI measurements |
| Neck angle | 45–135° | Rotational DSA |

---

## Surrogate Architectures

### 1. Gaussian Process Regression (`GPSurrogate`)
- Kernel: `ConstantKernel × ARD-RBF + WhiteKernel`  
- Per-output GPs with calibrated predictive uncertainty  
- Best for small high-fidelity datasets (~300 samples)
- Provides **error bars** critical for regulatory UQ

### 2. Neural Network (`NNSurrogate`)  
- Architecture: `[64 → 128 → 64]` ReLU, Adam, early stopping
- Handles full dataset (1600 samples); fastest inference
- Uncertainty via input-perturbation ensemble (lightweight MC approximation)

### 3. Polynomial Chaos Expansion (`PCESurrogate`)
- Degree-2 Legendre basis; least-squares fit  
- **Analytical global sensitivity analysis** (first-order Sobol indices)
- Standard in FDA/ISO UQ workflows (OpenTURNS, Dakota compatible interface)
- Most interpretable: maps directly to regulatory sensitivity reports

---

## Results

Evaluated on 400-sample holdout (20% of 2000-sample dataset):

| Model | Mean R² | Normalised RMSE | Inference time |
|-------|---------|-----------------|----------------|
| GP    | **0.976** | lowest per-output | ~32 µs/sample |
| NN    | 0.973   | medium          | ~7 µs/sample  |
| PCE   | 0.943   | higher (degree-2) | ~5 µs/sample  |

All three surrogates achieve R² > 0.94 across all 7 haemodynamic QoIs.

### Key figures

| Figure | Content |
|--------|---------|
| `fig1_accuracy_comparison.png` | R² and normalised RMSE per output, all models |
| `fig2_parity_rupture_risk.png` | Parity plots for rupture risk score (most clinically relevant QoI) |
| `fig3_gp_uncertainty.png` | GP calibration: `\|error\| vs σ` + prediction intervals |
| `fig4_sobol_sensitivity.png` | PCE global sensitivity heatmap (all inputs × all outputs) |
| `fig5_speedup.png` | Inference speedup vs CFD oracle |

---

## Project Structure

```
vascular-surrogate/
├── cfd_simulator.py       # Physics-based CFD oracle + dataset generator
├── surrogate_models.py    # GP, NN, PCE surrogate classes
├── run_analysis.py        # Full benchmark pipeline + figures
├── figures/               # Generated plots
└── README.md
```

---

## Usage

```python
from cfd_simulator import AneurysmGeometry, run_cfd, generate_dataset
from surrogate_models import GPSurrogate, NNSurrogate, PCESurrogate

# Single CFD evaluation
geom = AneurysmGeometry(neck_diameter=0.004, sac_diameter=0.009,
                        parent_diameter=0.004, inflow_velocity=0.40)
result = run_cfd(geom)
print(f"Rupture risk: {result.rupture_risk_score:.3f}")
print(f"Max WSS: {result.max_wss:.2f} Pa")

# Generate dataset and train surrogate
X, y, feat_names, tgt_names = generate_dataset(n=1000)
model = NNSurrogate()
model.fit(X[:800], y[:800])
scores = model.score(X[800:], y[800:])
print(f"R² = {scores['r2'].mean():.4f}")

# Global sensitivity via PCE
pce = PCESurrogate(degree=2).fit(X[:800], y[:800])
sobol = pce.sobol_first_order()   # shape: [9 inputs, 7 outputs]
```

Run full benchmark:
```bash
python run_analysis.py
```

---

## Dependencies

```
numpy scipy scikit-learn matplotlib
```

Install:
```bash
pip install numpy scipy scikit-learn matplotlib
```

---

## Connection to In-Silico Trials

This project directly addresses the computational bottleneck in regulatory in-silico trial frameworks (e.g., FDA 2023 *Reporting of Computational Modelling Studies*). The three surrogate types cover the standard UQ toolkit:

- **GP**: small-data regime, calibrated uncertainty, Bayesian experimental design  
- **NN**: large-data regime, fast Monte Carlo propagation  
- **PCE**: regulatory-grade sensitivity analysis, compatible with OpenTURNS/Dakota  

Planned extensions: physics-informed neural networks (PINNs), reduced-order models (ROM via POD), geometry-aware graph neural networks for patient-specific meshes.

---

## References

- Shojima et al. (2004). *Magnitude and role of wall shear stress on cerebral aneurysm.* Stroke 35:2500–2505.
- Sforza et al. (2009). *Hemodynamics of cerebral aneurysms.* Ann Biomed Eng 37:1–30.
- Xiang et al. (2011). *Hemodynamic–morphological discriminants for intracranial aneurysm rupture.* Stroke 42:144–152.
- FDA (2023). *Reporting of computational modelling studies in medical device submissions.*
- Sudret (2008). *Global sensitivity analysis using polynomial chaos expansions.* Reliab Eng Syst Saf.
