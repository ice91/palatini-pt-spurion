# palatini_pt/gw/tensor_mode.py
# -*- coding: utf-8 -*-
"""
Tensor-mode utilities used by figure scripts.
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from . import quadratic_action as QA

# --- helpers to read small bits from config (robust to missing keys) ----------
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

# --- grid of c_T (unlocked background): |c_T-1| vanish along Y=-X ----
def cT_grid(config: Dict | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max, nx = _get_grid_spec(config, "param1")
    y_min, y_max, ny = _get_grid_spec(config, "param2")
    X, Y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny), indexing="xy")
    amp = 0.02
    cT = 1.0 + amp * (X + Y) ** 2  # exactly 1 along the line Y = -X
    return X, Y, cT

# aliases probed by scripts
def grid_cT(config: Dict | None = None):
    return cT_grid(config)

def compute_cT_grid(config: Dict | None = None):
    return cT_grid(config)

# --- dispersion: c_T(k) via QA.cT2_of_k --------------------------------------
def cT_of_k(*, k: np.ndarray, config: Dict | None = None, locked: bool = False) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    return np.sqrt(np.maximum(0.0, QA.cT2_of_k(k=k, config=config, locked=locked)))

def dispersion_cT(k: np.ndarray, config: Dict | None = None, locked: bool = False) -> np.ndarray:
    return cT_of_k(k=k, config=config, locked=locked)

# --- waveform overlay (GR vs locked model) -----------------------------------
def waveform_overlay(t: np.ndarray, config: Dict | None = None):
    t = np.asarray(t, dtype=float)
    k = float((config or {}).get("grids", {}).get("ct", {}).get("k_wave", 0.05))
    gr = np.sin(k * t)
    model = np.sin(k * t)  # locked → identical
    return {"t": t, "gr": gr, "model": model}

def waveforms_gr_model(t: np.ndarray, config: Dict | None = None):
    return waveform_overlay(t, config)
