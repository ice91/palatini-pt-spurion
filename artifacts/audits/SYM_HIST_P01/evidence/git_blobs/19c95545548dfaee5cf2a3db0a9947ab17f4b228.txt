# -*- coding: utf-8 -*-
"""
palatini_pt.gw.quadratic_action
================================

最小可用版的張量擾動二次作用量封裝。核心假設（為了 C3 測試與出圖）：

- 二次作用量可化為
    S_T^(2) ~ ∫ dτ d^3k  (Z/2) [ (h')^2 - c_T^2 k^2 h^2 ],
  其中 Z>0 是動力學係數、c_T^2 是張量傳播速度平方。
- 在本專案的 O(∂^2) 階次下，c_T^2 的偏離由共同基底係數向量的一個
  **固定線性組合** Δ = L·c 決定：
    c_T^2 = 1 + Δ
  這樣一來，若滿足 "locking" 條件（L·c = 0），則可保證對所有 k 有 c_T ≡ 1。

此模組僅依賴 numpy，並對 "係數表" 使用 Mapping[str, float] 介面。
實務上，係數表由 equivalence/coeff_extractor.py 投影到共同基底後取得。

Public API
----------
- get_ct2_from_coeffs(coeffs) -> float
- get_Z_from_coeffs(coeffs) -> float
- quadratic_action_params(coeffs) -> dict(Z=float, cT2=float)
- locking_weights() -> dict[str, float]  # 與 locking 模組共用的 L 向量
"""
from __future__ import annotations

from typing import Mapping, Dict

import numpy as np


# 與 locking 邏輯共享的 c_T^2 線性組合權重（可視為定義基底下的 L_i）
# 你可以在將來改成讀自 algebra/basis 或 equivalence 的明細表。
# 這版選擇以最小基底中常見的三個 monomials 作為示意：
#   - "grad_eps2"                 ≈ (∂ε)^2
#   - "box_eps"                  ≈ □ε
#   - "torsion_trace_grad_eps"   ≈ T_μ ∂^μ ε
# 注意：若某鍵不存在，函式會把該係數視為 0。
_L_WEIGHTS: Dict[str, float] = {
    "grad_eps2": 0.25,
    "box_eps": -0.5,
    "torsion_trace_grad_eps": 0.125,
}


def locking_weights() -> Dict[str, float]:
    """
    回傳控制 c_T^2 偏離的線性權重 L_i。

    Returns
    -------
    dict
        形如 {"grad_eps2": 0.25, "box_eps": -0.5, ...}
    """
    return dict(_L_WEIGHTS)


def _coef(coeffs: Mapping[str, float], key: str) -> float:
    """安全取得係數（缺鍵視為 0.0）。"""
    v = coeffs.get(key, 0.0)
    # 容忍 numpy scalar / python scalar
    return float(v)


def get_ct2_from_coeffs(coeffs: Mapping[str, float]) -> float:
    """
    由係數表計算 c_T^2 = 1 + Δ，其中 Δ = Σ_i L_i c_i。

    Parameters
    ----------
    coeffs : Mapping[str, float]
        共同基底下的係數表（可只給部分鍵，缺省視為 0）。

    Returns
    -------
    float
        c_T^2
    """
    delta = 0.0
    for k, w in _L_WEIGHTS.items():
        delta += w * _coef(coeffs, k)
    return 1.0 + delta


def get_Z_from_coeffs(coeffs: Mapping[str, float]) -> float:
    """
    動力學係數 Z 的最小建模：保證正且隨 (∂ε)^2 與 |□ε| 略增，避免病態。
    你可以將來改成真實推導的解析式。

    Z = 1 + 0.10*|grad_eps2| + 0.05*|box_eps|

    Returns
    -------
    float
        Z > 0
    """
    ge2 = abs(_coef(coeffs, "grad_eps2"))
    bxe = abs(_coef(coeffs, "box_eps"))
    Z = 1.0 + 0.10 * ge2 + 0.05 * bxe
    # 若使用者誤給到極端負係數，仍保底到 1e-12 避免奇異
    return float(max(Z, 1e-12))


def quadratic_action_params(coeffs: Mapping[str, float]) -> Dict[str, float]:
    """
    封裝（Z, c_T^2）兩個二次作用量參數。

    Parameters
    ----------
    coeffs : Mapping[str, float]
        共同基底係數表

    Returns
    -------
    dict
        {"Z": float, "cT2": float}
    """
    return {"Z": get_Z_from_coeffs(coeffs), "cT2": get_ct2_from_coeffs(coeffs)}
