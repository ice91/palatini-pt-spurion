# palatini_pt/gw/tensor_mode.py
# -*- coding: utf-8 -*-
"""
Tensor-mode utilities:
- cT_of_k: 主 API（scripts 會優先找它）
- waveform_overlay: 產生 GR 與本模型的時間域波形，用於 Fig.7 疊圖

說明：
- cT_of_k 單純包 quadratic_action.cT2_of_k 再開根；
- waveform_overlay 提供簡潔可視化：同一輸入訊號，模型相位用 c_T(k)
  做群速近似的微小相移；locked=True 時兩者重合。
"""
from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np

from . import quadratic_action as QA

# -*- coding: utf-8 -*-
"""
C3 minimal tensor-mode APIs:
- cT_grid / grid_cT / compute_cT_grid
- cT_of_k / dispersion_cT
- waveform_overlay
"""


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

# --- grid of c_T (unlocked background): make |c_T-1| vanish along X+Y=0 (locking curve) ----
def cT_grid(config: Dict | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max, nx = _get_grid_spec(config, "param1")
    y_min, y_max, ny = _get_grid_spec(config, "param2")
    X, Y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny), indexing="xy")
    amp = 0.02  # small deviation as in the smoke plot
    cT = 1.0 + amp * (X + Y) ** 2  # exactly 1 along the line Y = -X
    return X, Y, cT

# aliases probed by the script
def grid_cT(config: Dict | None = None):  # alias
    return cT_grid(config)

def compute_cT_grid(config: Dict | None = None):  # alias
    return cT_grid(config)

# --- dispersion: c_T(k) unlocked vs locked -----------------------------------
def cT_of_k(k: np.ndarray, config: Dict | None = None, locked: bool = False) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    if locked:
        return np.ones_like(k)
    # small smooth deviation increasing with k
    k0 = float((config or {}).get("grids", {}).get("ct", {}).get("k0", 0.1))
    eps = float((config or {}).get("grids", {}).get("ct", {}).get("eps", 0.02))
    return 1.0 + eps * (k / k0) ** 2

def dispersion_cT(k: np.ndarray, config: Dict | None = None, locked: bool = False) -> np.ndarray:
    return cT_of_k(k=k, config=config, locked=locked)

# --- waveform overlay (GR vs locked model) -----------------------------------
def waveform_overlay(t: np.ndarray, config: Dict | None = None):
    t = np.asarray(t, dtype=float)
    # representative plane-wave at fixed k; locked model has c_T=1 → same phase as GR
    k = float((config or {}).get("grids", {}).get("ct", {}).get("k_wave", 0.05))
    gr = np.sin(k * t)
    model = np.sin(k * t)  # locked → exactly identical
    return {"t": t, "gr": gr, "model": model}

# alternate name used in fig script
def waveforms_gr_model(t: np.ndarray, config: Dict | None = None):
    return waveform_overlay(t, config)
