# palatini_pt/gw/quadratic_action.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
import numpy as np

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

# 提供一個不會造成相依循環的別名，供部分程式碼呼叫
def cT_squared(k: np.ndarray, config: Dict | None = None, locked: bool = False) -> np.ndarray:
    return cT2_of_k(k=k, config=config, locked=locked)
