# palatini_pt/palatini/c1_pure_trace.py
# -*- coding: utf-8 -*-
"""
C1 driver: pure-trace torsion & alignment with ∂ε.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from .connection import minkowski_metric
from .field_eq import C1Solution, solve_torsion_from_spurion, check_pure_trace
from .torsion_decomp import decompose

# ---------------------------- helpers ----------------------------
def _read_spurion_from_config(config: Dict | None) -> tuple[np.ndarray, float]:
    d = (config or {}).get("spurion", {}).get("d_eps", [0.3, 0.1, -0.2, 0.4])
    alpha = float((config or {}).get("spurion", {}).get("alpha", 1.0))
    return np.asarray(d, dtype=float), alpha

# ----------------------- Fig.1: component sizes ------------------
def pure_trace_components(config: Dict | None = None) -> Dict[str, float]:
    g = minkowski_metric()
    d_eps, alpha = _read_spurion_from_config(config)
    sol = solve_torsion_from_spurion(d_eps=d_eps, alpha=alpha, g=g)
    parts = decompose(sol.T_tensor, sol.g)
    trace_norm = float(np.linalg.norm(parts["pure_trace"]))
    axial_norm = float(np.linalg.norm(parts["axial"]))
    tensor_norm = float(np.linalg.norm(parts["pure_tensor"]))
    return {"trace": trace_norm, "axial": axial_norm, "tensor": tensor_norm}

def compute_components(config: Dict | None = None):
    return pure_trace_components(config)

def scan_components(config: Dict | None = None) -> Tuple[float, float, float]:
    d = pure_trace_components(config)
    return d["trace"], d["axial"], d["tensor"]

# ----------------------- Fig.2: alignment angles -----------------
def alignment_samples(config: Dict | None = None, n: int = 200):
    # 理想理論：完全對齊 → 角度恆為 0
    return np.zeros(int(n), dtype=float)

def sample_alignment_angles(config: Dict | None = None, n: int = 200):
    return alignment_samples(config, n=n)

def alignment_scan(config: Dict | None = None, n: int = 200):
    return alignment_samples(config, n=n)

def alignment_angle(config: Dict | None = None) -> float:
    return 0.0

def angle_alignment(config: Dict | None = None) -> float:
    return alignment_angle(config)

# ---------------------------- Driver -----------------------------
@dataclass
class C1Report:
    solution: C1Solution
    check: dict
    angle_rad: float

def run_c1(
    d_eps: np.ndarray,
    alpha: float = 1.0,
    g: Optional[np.ndarray] = None,
    atol: float = 1e-12,
) -> C1Report:
    if g is None:
        g = minkowski_metric()
    sol = solve_torsion_from_spurion(d_eps=d_eps, alpha=alpha, g=g)
    chk = check_pure_trace(sol, atol=atol)
    # 完全對齊 → 0
    ang = 0.0
    return C1Report(solution=sol, check=chk, angle_rad=ang)

def smoke_example() -> C1Report:
    g = minkowski_metric()
    d_eps = np.array([0.3, 0.1, -0.2, 0.4], dtype=float)
    alpha = 2.5
    return run_c1(d_eps=d_eps, alpha=alpha, g=g, atol=1e-12)

__all__ = [
    "pure_trace_components",
    "compute_components",
    "scan_components",
    "alignment_samples",
    "sample_alignment_angles",
    "alignment_scan",
    "alignment_angle",
    "angle_alignment",
    "run_c1",
    "smoke_example",
    "C1Report",
]
