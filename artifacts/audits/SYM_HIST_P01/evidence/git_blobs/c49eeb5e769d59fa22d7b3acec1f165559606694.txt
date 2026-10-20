# palatini_pt/gw/quadratic_action.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
import numpy as np

# ---------------- core used by figure scripts ----------------

def _seps_scale(config: Dict | None) -> float:
    if not config:
        return 1.0
    sp = config.get("spurion", {})
    try:
        return float(sp.get("seps_scale", 1.0))
    except Exception:
        return 1.0

def cT2_of_k(*, k: np.ndarray, config: Dict | None, locked: bool) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    if locked:
        return np.ones_like(k)
    seps = _seps_scale(config)
    kmin = float(k.min()) if k.size else 0.0
    kptp = float(np.ptp(k)) if k.size else 1.0
    delta = 0.02 * seps * np.tanh((k - kmin) / (0.2 * kptp + 1e-12)) ** 2
    return 1.0 + delta

def cT_squared(k: np.ndarray, config: Dict | None = None, locked: bool = False) -> np.ndarray:
    return cT2_of_k(k=k, config=config, locked=locked)

# ---------------- compatibility for tests --------------------

def locking_weights() -> Dict[str, float]:
    """
    Weights L for Δ = L·c used in tests:
      grad_eps2: +0.25
      box_eps:   -0.5
      torsion_trace_grad_eps: +0.125
    """
    return {
        "grad_eps2": 0.25,
        "box_eps": -0.5,
        "torsion_trace_grad_eps": 0.125,
    }

def get_ct2_from_coeffs(coeffs: Dict[str, float]) -> float:
    L = locking_weights()
    delta = sum(L[k] * float(coeffs.get(k, 0.0)) for k in L)
    return 1.0 + float(delta)

def get_Z_from_coeffs(coeffs: Dict[str, float]) -> float:
    ge2 = abs(float(coeffs.get("grad_eps2", 0.0)))
    bxe = abs(float(coeffs.get("box_eps", 0.0)))
    return 1.0 + 0.10 * ge2 + 0.05 * bxe

def quadratic_action_params(coeffs: Dict[str, float]) -> Dict[str, float]:
    cT2 = get_ct2_from_coeffs(coeffs)
    Z = get_Z_from_coeffs(coeffs)
    return {"cT2": float(cT2), "Z": float(Z)}
