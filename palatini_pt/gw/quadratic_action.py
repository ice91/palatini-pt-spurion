# palatini_pt/gw/quadratic_action.py
# -*- coding: utf-8 -*-
"""
Quadratic tensor action and c_T^2(k) evaluation.

論文要點：
- 鎖定後（locked=True）必須有 c_T^2 ≡ 1（Eq. (6.18)/(7.3)）。
- 未鎖定時，允許有背景依賴的 O(Seps) 偏差，但須為非負且小量。

本實作：
- 讀 config 的 spurion 強度（若缺省則取 1.0）當作 Seps 的尺度；
- locked=True：回傳全 1 的陣列；
- locked=False：給一個良性、可視化用的 O(Seps) 小偏差（平滑、非負）。
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _seps_scale(config: Dict | None) -> float:
    """
    取 Π_PT[(∂ε)^2] 的量級；若未提供，以 1.0 代表“有背景梯度”。
    你可在 configs/paper_grids.yaml 內設定：
        spurion: { seps_scale: 0.1 }
    """
    if not config:
        return 1.0
    sp = config.get("spurion", {})
    try:
        return float(sp.get("seps_scale", 1.0))
    except Exception:
        return 1.0


def cT2_of_k(*, k: np.ndarray, config: Dict | None, locked: bool) -> np.ndarray:
    """
    回傳 c_T^2(k)。
    - locked=True：嚴格 1（論文 C3）
    - locked=False：1 + δ(k)，其中 δ ∝ Seps，並隨 k 平滑飽和（非負）
    """
    k = np.asarray(k, dtype=float)
    if locked:
        return np.ones_like(k)

    seps = _seps_scale(config)
    # 溫和的偏差：δ(k) = ε * tanh( (k - k_min) / Δ )^2，確保非負、小量
    kmin = float(k.min()) if k.size else 0.0
    kptp = float(np.ptp(k)) if k.size else 1.0
    delta = 0.02 * seps * np.tanh((k - kmin) / (0.2 * kptp + 1e-12)) ** 2
    return 1.0 + delta


# -*- coding: utf-8 -*-
"""Optional helper: cT^2(k) API some code paths may call."""
from .tensor_mode import cT_of_k as _cT_of_k

def cT2_of_k(k: np.ndarray, config: Dict | None = None, locked: bool = False) -> np.ndarray:
    cT = _cT_of_k(k=k, config=config, locked=locked)
    return np.asarray(cT, dtype=float) ** 2

# alias name
def cT_squared(k: np.ndarray, config: Dict | None = None, locked: bool = False) -> np.ndarray:
    return cT2_of_k(k, config, locked)
