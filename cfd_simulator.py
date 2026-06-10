"""
cfd_simulator.py
----------------
Synthetic "CFD oracle" for idealized intracranial aneurysm geometries.

Physics basis:
  - Steady-state Navier-Stokes in a simplified toroidal sac geometry
  - Wall shear stress (WSS) derived analytically from Poiseuille-like flow
    modified by sac aspect ratio and neck diameter (Shojima 2004 parameterisation)
  - Oscillatory Shear Index (OSI) estimated from pulsatility + geometry
  - Pressure drop via Bernoulli with viscous correction
  - Inflow velocity from Womersley number scaling

This module is the "expensive solver" the surrogate replaces.
Each call represents ~hours of real CFD wall-time; here it runs in ~ms
but uses the same input/output interface as a real solver would.

References (conceptual):
  Shojima et al. (2004) Stroke 35:2500–2505  (WSS in cerebral aneurysms)
  Sforza et al. (2009) Ann Biomed Eng 37:1–30 (aneurysm haemodynamics review)
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Physical constants (blood, SI units)
# ---------------------------------------------------------------------------
RHO   = 1060.0   # kg/m³  blood density
MU    = 3.5e-3   # Pa·s   dynamic viscosity
HEART_RATE_HZ = 1.2   # Hz    resting ~72 bpm


@dataclass
class AneurysmGeometry:
    """
    Parameterisation of an idealized saccular intracranial aneurysm.

    All lengths in metres.  Distributions taken from clinical literature
    (ISUIA cohort + morphological studies).

    Parameters
    ----------
    neck_diameter   : diameter of the aneurysm neck [m]
    sac_diameter    : maximum sac diameter [m]
    parent_diameter : parent artery diameter [m]
    aspect_ratio    : sac_height / neck_diameter  (computed if not given)
    inflow_velocity : mean inflow velocity [m/s]
    angle_deg       : parent-artery to aneurysm neck angle [degrees]
    """
    neck_diameter:   float = 0.003    # 3 mm
    sac_diameter:    float = 0.007    # 7 mm
    parent_diameter: float = 0.004    # 4 mm
    inflow_velocity: float = 0.35     # m/s  (ICA typical)
    angle_deg:       float = 90.0     # degrees

    @property
    def aspect_ratio(self) -> float:
        """Height / neck — classic rupture-risk morphological index."""
        return self.sac_diameter / self.neck_diameter

    @property
    def size_ratio(self) -> float:
        return self.sac_diameter / self.parent_diameter

    @property
    def reynolds(self) -> float:
        return RHO * self.inflow_velocity * self.parent_diameter / MU

    @property
    def womersley(self) -> float:
        import math
        omega = 2 * math.pi * HEART_RATE_HZ
        return self.parent_diameter / 2 * np.sqrt(RHO * omega / MU)

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.neck_diameter,
            self.sac_diameter,
            self.parent_diameter,
            self.inflow_velocity,
            self.angle_deg,
            self.aspect_ratio,
            self.size_ratio,
            self.reynolds,
            self.womersley,
        ], dtype=np.float64)

    @classmethod
    def feature_names(cls):
        return [
            "neck_diameter", "sac_diameter", "parent_diameter",
            "inflow_velocity", "angle_deg",
            "aspect_ratio", "size_ratio",
            "reynolds", "womersley",
        ]


@dataclass
class CFDResult:
    """Scalar QoIs extracted from a CFD solution."""
    max_wss:      float   # Pa   — max wall shear stress in sac
    mean_wss:     float   # Pa   — mean wall shear stress
    min_wss:      float   # Pa   — low-WSS area correlates with rupture
    osi:          float   # []   — oscillatory shear index  [0, 0.5]
    pressure_drop:float   # Pa   — neck-to-dome pressure difference
    vortex_strength: float  # []  — normalised vortex core strength proxy
    rupture_risk_score: float  # composite [0,1] (post-processing index)


def run_cfd(geom: AneurysmGeometry,
            noise_sigma: float = 0.03,
            seed: Optional[int] = None) -> CFDResult:
    """
    Physics-based analytical approximation of CFD QoIs.

    Equations
    ---------
    WSS_parent = 4 * mu * Q / (pi * r_parent^3)   [Poiseuille]
    WSS_neck   = WSS_parent * (r_parent/r_neck)^3  [flow conservation]
    WSS_max    = WSS_neck * f(aspect_ratio, angle)
    OSI        ~ 0.5 * (1 - 1/(1 + 0.1*Wo*AR))    [Womersley scaling]
    dP         = 8*mu*L*Q / (pi*r^4)  + rho*v^2/2 [Bernoulli+viscous]

    Gaussian noise is added to model the variability between mesh resolutions.
    """
    rng = np.random.default_rng(seed)

    r_p = geom.parent_diameter / 2
    r_n = geom.neck_diameter / 2
    Q = np.pi * r_p**2 * geom.inflow_velocity   # volumetric flow [m³/s]

    # --- WSS in parent artery (Poiseuille baseline) ---
    wss_parent = 4 * MU * Q / (np.pi * r_p**3)

    # --- WSS at neck (continuity scaling) ---
    wss_neck = wss_parent * (r_p / r_n)**3

    # --- Geometric amplification inside sac ---
    AR = geom.aspect_ratio
    theta = np.radians(geom.angle_deg)
    # High AR → impingement → elevated max WSS near neck
    # Low  AR → recirculation → low WSS over dome
    amp = 1.0 + 0.6 * np.exp(-AR / 2.5) * np.abs(np.sin(theta))
    wss_max  = wss_neck * amp * (1 + rng.normal(0, noise_sigma))
    wss_mean = wss_neck * 0.45 * (1 + rng.normal(0, noise_sigma))
    wss_min  = wss_mean * 0.12 * np.exp(-AR / 3) * (1 + rng.normal(0, noise_sigma))

    # --- OSI (oscillatory shear index) ---
    Wo = geom.womersley
    osi = 0.5 * (1 - 1 / (1 + 0.08 * Wo * AR))
    osi = np.clip(osi + rng.normal(0, 0.01), 0.0, 0.5)

    # --- Pressure drop across neck ---
    L_eff = geom.sac_diameter          # effective flow path
    dP_viscous = 8 * MU * L_eff * Q / (np.pi * r_n**4)
    dP_dynamic = 0.5 * RHO * geom.inflow_velocity**2 * (1 - (r_n/r_p)**4)
    pressure_drop = (dP_viscous + dP_dynamic) * (1 + rng.normal(0, noise_sigma))

    # --- Vortex strength proxy ---
    Re = geom.reynolds
    vortex = np.tanh(Re / 400 * AR / 2) * (1 + rng.normal(0, noise_sigma * 0.5))
    vortex = float(np.clip(vortex, 0, 1))

    # --- Composite rupture risk (literature-derived weights) ---
    # Based on Xiang et al. 2011: low WSS + high OSI = highest risk
    risk = (
        0.4 * np.clip(1 - wss_min / 2.0, 0, 1)  # low WSS → risk
      + 0.3 * (osi / 0.5)                         # high OSI → risk
      + 0.2 * np.clip(AR / 3, 0, 1)               # high AR → risk
      + 0.1 * np.clip(geom.size_ratio / 3, 0, 1)  # large sac → risk
    )
    risk = float(np.clip(risk + rng.normal(0, 0.02), 0, 1))

    return CFDResult(
        max_wss=float(wss_max),
        mean_wss=float(wss_mean),
        min_wss=float(wss_min),
        osi=float(osi),
        pressure_drop=float(pressure_drop),
        vortex_strength=vortex,
        rupture_risk_score=risk,
    )


def generate_dataset(n: int = 2000, seed: int = 42) -> tuple:
    """
    Latin Hypercube-style sampling across the clinical parameter space.

    Parameter ranges from ISUIA + morphological literature:
      neck:    2–8 mm
      sac:     3–15 mm  (must be >= neck)
      parent:  2–5 mm
      inflow:  0.2–0.6 m/s
      angle:   45–135 deg
    """
    rng = np.random.default_rng(seed)

    necks   = rng.uniform(0.002, 0.008, n)
    parents = rng.uniform(0.002, 0.005, n)
    sacs    = necks + rng.uniform(0.001, 0.010, n)   # sac >= neck always
    inflows = rng.uniform(0.20, 0.60, n)
    angles  = rng.uniform(45.0, 135.0, n)

    X_list, y_list = [], []
    for i in range(n):
        g = AneurysmGeometry(
            neck_diameter=necks[i],
            sac_diameter=sacs[i],
            parent_diameter=parents[i],
            inflow_velocity=inflows[i],
            angle_deg=angles[i],
        )
        result = run_cfd(g, seed=seed + i)
        X_list.append(g.to_feature_vector())
        y_list.append([
            result.max_wss, result.mean_wss, result.min_wss,
            result.osi, result.pressure_drop,
            result.vortex_strength, result.rupture_risk_score,
        ])

    X = np.array(X_list)
    y = np.array(y_list)
    feature_names = AneurysmGeometry.feature_names()
    target_names  = ["max_wss", "mean_wss", "min_wss",
                     "osi", "pressure_drop",
                     "vortex_strength", "rupture_risk_score"]
    return X, y, feature_names, target_names
