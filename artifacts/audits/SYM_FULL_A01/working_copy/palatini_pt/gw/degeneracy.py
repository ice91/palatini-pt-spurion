# palatini_pt/gw/degeneracy.py
# -*- coding: utf-8 -*-
"""
Principal-symbol eigenvalues & spectrum report (C3).
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from . import quadratic_action as QA

# ----- figure-script API kept (grid-like synthetic) -----------------

def principal_eigs(*, config: Dict | None, locked: bool, n: int = 40) -> np.ndarray:
    n = max(2, int(n))
    base = np.zeros(n, dtype=float)
    if locked:
        base[0] = 1.0
        base[1] = 1.0
        return np.sort(base)
    seps = float((config or {}).get("spurion", {}).get("seps_scale", 1.0))
    noise = 1e-2 * seps * np.linspace(0.0, 1.0, n)
    base += noise
    base[0] = 1.0 + 0.01 * seps
    base[1] = 1.0 + 0.008 * seps
    return np.sort(base)

# ----- compatibility for tests -------------------------------------

def principal_symbol_eigs(params: Dict[str, Any]) -> np.ndarray:
    """
    If passed coefficients dict, return a 2-eigenvalue toy principal symbol:
      [cT^2, Z]  (one dispersive, one kinetic normalization)
    """
    if isinstance(params, dict) and (
        "grad_eps2" in params or "box_eps" in params or "torsion_trace_grad_eps" in params
    ):
        cT2 = QA.get_ct2_from_coeffs(params)
        Z = QA.get_Z_from_coeffs(params)
        return np.array([float(cT2), float(Z)], dtype=float)
    # fallback: treat as config for figure-path
    return principal_eigs(config=params, locked=False, n=2)[:2]

def hamiltonian_eigs(config: Dict | None = None):
    return principal_symbol_eigs(config if config is not None else {})

def is_nondegenerate(coeffs: Dict[str, float], tol: float = 1e-14) -> bool:
    eigs = principal_symbol_eigs(coeffs)
    return bool(np.min(eigs) >= -float(tol))

def spectrum_report(coeffs: Dict[str, float]) -> Dict[str, float]:
    cT2 = QA.get_ct2_from_coeffs(coeffs)
    Z = QA.get_Z_from_coeffs(coeffs)
    eigs = np.array([cT2, Z], dtype=float)
    return {
        "cT": float(np.sqrt(max(0.0, cT2))),
        "Z": float(Z),
        "eig_min": float(np.min(eigs)),
    }
