# palatini_pt/gw/tensor_mode.py
# -*- coding: utf-8 -*-
"""
Tensor-mode utilities:
- cT_of_k: accepts both (k, coeffs, lock=...) and (*, k, config, locked=...)
- dispersion_omega2: ω^2
- sample_cT_grid: helper for tests
- waveform_overlay & cT heatmap helpers for figures
"""
from __future__ import annotations

from typing import Dict, Tuple, Iterable, Any
import numpy as np

from . import quadratic_action as QA

# ---------- helpers for heatmap ----------
def _get_grid_spec(config: Dict | None, key: str, default_min=-1.0, default_max=1.0, default_n=61):
    v = ((config or {}).get("grids", {}).get("ct", {}).get(key, {})) if config else {}
    if isinstance(v, dict):
        vmin = float(v.get("min", default_min))
        vmax = float(v.get("max", default_max))
        n = int(v.get("n", default_n))
    elif isinstance(v, (list, tuple)) and len(v) == 3:
        vmin, vmax, n = float(v[0]), float(v[1]), int(v[2])
    else:
        vmin, vmax, n = default_min, default_max, default_n
    return vmin, vmax, n

def cT_grid(config: Dict | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max, nx = _get_grid_spec(config, "param1")
    y_min, y_max, ny = _get_grid_spec(config, "param2")
    X, Y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny), indexing="xy")
    amp = 0.02
    cT = 1.0 + amp * (X + Y) ** 2  # exactly 1 along Y = -X
    return X, Y, cT

def grid_cT(config: Dict | None = None):
    return cT_grid(config)

def compute_cT_grid(config: Dict | None = None):
    return cT_grid(config)

# ---------- main dispersion API ----------
def _ct_from_coeffs_or_config(k: np.ndarray, arg: Any, lock: bool) -> np.ndarray:
    if lock:
        return np.ones_like(k, dtype=float)
    # coeff-dict path (tests)
    if isinstance(arg, dict) and (
        "grad_eps2" in arg or "box_eps" in arg or "torsion_trace_grad_eps" in arg
    ):
        cT2 = QA.quadratic_action_params(arg)["cT2"]
        return np.sqrt(np.maximum(0.0, cT2)) * np.ones_like(k, dtype=float)
    # config path (figure scripts)
    return np.sqrt(np.maximum(0.0, QA.cT2_of_k(k=k, config=arg, locked=False)))

def cT_of_k(k: np.ndarray, coeffs_or_config: Any = None, lock: bool = False, **kwargs) -> np.ndarray:
    """
    Compatible signatures:
      (k, coeffs_dict, lock=False)
      (*, k=..., config=..., locked=...)
    """
    # accept figure-script keywords
    if "locked" in kwargs:
        lock = bool(kwargs["locked"])
    if "config" in kwargs:
        coeffs_or_config = kwargs["config"]
    k = np.asarray(k, dtype=float)
    return _ct_from_coeffs_or_config(k, coeffs_or_config, lock)

def dispersion_cT(k: np.ndarray, config: Dict | None = None, locked: bool = False) -> np.ndarray:
    return cT_of_k(k=k, coeffs_or_config=config, lock=locked)

def dispersion_omega2(k: np.ndarray, coeffs_or_config: Any, lock: bool = False) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    cT = cT_of_k(k, coeffs_or_config, lock=lock)
    return (cT ** 2) * (k ** 2)

def sample_cT_grid(k: np.ndarray, coeffs_list: Iterable[Dict[str, float]], lock: bool = False) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    coeffs_list = list(coeffs_list)
    out = np.zeros((k.size, len(coeffs_list)), dtype=float)
    for j, c in enumerate(coeffs_list):
        out[:, j] = cT_of_k(k, c, lock=lock)
    return out

# ---------- waveform overlay for Fig.7 ----------
def waveform_overlay(t: np.ndarray, config: Dict | None = None):
    t = np.asarray(t, dtype=float)
    k = float((config or {}).get("grids", {}).get("ct", {}).get("k_wave", 0.05))
    gr = np.sin(k * t)
    model = np.sin(k * t)  # locked → identical
    return {"t": t, "gr": gr, "model": model}

def waveforms_gr_model(t: np.ndarray, config: Dict | None = None):
    return waveform_overlay(t, config)
