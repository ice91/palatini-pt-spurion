# palatini_pt/gw/locking.py
# -*- coding: utf-8 -*-
"""
Coefficient "locking" utilities (C3) used by tests and figure overlays.
Also provides a simple locking-curve for the heatmap overlay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Iterable, Tuple
import numpy as np

from .quadratic_action import locking_weights

# ---- minimal "paper-figure" helpers (kept) ---------------------------------

def apply(*, coeffs: Dict[str, float] | Dict[str, Any]) -> Dict[str, Any]:
    out = dict(coeffs) if coeffs is not None else {}
    out["locked"] = True
    return out

def locking_curve(config: Dict | None = None):
    gx = ((config or {}).get("grids", {}).get("ct", {}).get("param1", {})) if config else {}
    x_min = float(getattr(gx, "get", lambda *_: -1.0)("min", -1.0))
    x_max = float(getattr(gx, "get", lambda *_:  1.0)("max",  1.0))
    n = int(getattr(gx, "get", lambda *_: 200)("n", 200))
    xs = np.linspace(x_min, x_max, n)
    ys = -xs
    return np.stack([xs, ys], axis=1)

def curve_locking(config: Dict | None = None):
    return locking_curve(config)

def locking_contour(config: Dict | None = None):
    return locking_curve(config)

# ---- full API expected by tests --------------------------------------------

def delta_ct2(coeffs: Dict[str, float]) -> float:
    L = locking_weights()
    return float(sum(L[k] * float(coeffs.get(k, 0.0)) for k in L))

def is_locked(coeffs: Dict[str, float], tol: float = 1e-10) -> bool:
    return abs(delta_ct2(coeffs)) <= float(tol)

def _choose_key_to_adjust(prefer_keys: Iterable[str] | None, coeffs: Dict[str, float]) -> str:
    L = locking_weights()
    if prefer_keys:
        for k in prefer_keys:
            if k in L:
                return k
    # default: prefer "box_eps" if present, else the first weight key
    if "box_eps" in L:
        return "box_eps"
    return next(iter(L.keys()))

def apply_locking(coeffs: Dict[str, float], prefer_keys: Iterable[str] | None = None) -> Dict[str, float]:
    L = locking_weights()
    d = delta_ct2(coeffs)
    if abs(d) == 0.0:
        return dict(coeffs)
    key = _choose_key_to_adjust(prefer_keys, coeffs)
    w = float(L[key])
    out = dict(coeffs)
    out[key] = float(out.get(key, 0.0)) - d / w
    return out

@dataclass
class LockReport:
    before_delta: float
    after_delta: float
    updated_coeffs: Dict[str, float]

def lock_and_report(coeffs: Dict[str, float], tol: float = 1e-10, prefer_keys: Iterable[str] | None = None) -> LockReport:
    d0 = delta_ct2(coeffs)
    updated = apply_locking(coeffs, prefer_keys=prefer_keys)
    d1 = delta_ct2(updated)
    return LockReport(before_delta=float(d0), after_delta=float(d1), updated_coeffs=updated)
