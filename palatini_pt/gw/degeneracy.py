# -*- coding: utf-8 -*-
"""
palatini_pt.gw.degeneracy
=========================

用「主值（principal symbol）」的極簡模型檢查 **不新增傳播 DoF**：

在二次作用量 S_T^(2) ~ (Z/2)[(h')^2 - c_T^2 k^2 h^2] 的最小設定下，
主值矩陣（忽略背景膨脹與耦合）可等效為對角：
    P = diag(Z, Z c_T^2)
其特徵值非負（Z≥0、Z c_T^2 ≥ 0）時，代表無額外不良傳播方向；
本模組給出：
- `principal_symbol_eigs(coeffs)`：回傳 [Z, Z*c_T^2]
- `is_nondegenerate(coeffs, tol=1e-14)`：是否非退化（均 ≥ -tol）
- `spectrum_report(coeffs)`：便利回傳字典（含 c_T 與是否非退化）
"""
from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from .quadratic_action import quadratic_action_params


def principal_symbol_eigs(coeffs: Mapping[str, float]) -> np.ndarray:
    """
    主值矩陣的兩個特徵值（對應兩個張量極化），最小模型為 [Z, Z*c_T^2]。

    Returns
    -------
    np.ndarray
        shape (2,)
    """
    pars = quadratic_action_params(coeffs)
    Z = float(pars["Z"])
    cT2 = float(pars["cT2"])
    return np.array([Z, Z * cT2], dtype=float)


def is_nondegenerate(coeffs: Mapping[str, float], tol: float = 1e-14) -> bool:
    """
    若兩個特徵值都 >= -tol，視為「非退化且無病態負向模式」。
    """
    eigs = principal_symbol_eigs(coeffs)
    return bool(np.all(eigs >= -float(tol)))


def spectrum_report(coeffs: Mapping[str, float]) -> Dict[str, float]:
    """
    小幫手：回傳 {Z, cT, eig_min, ok}。
    """
    pars = quadratic_action_params(coeffs)
    Z = float(pars["Z"])
    cT2 = float(pars["cT2"])
    cT = float(np.sqrt(max(cT2, 0.0)))
    eigs = principal_symbol_eigs(coeffs)
    return {
        "Z": Z,
        "cT": cT,
        "eig_min": float(np.min(eigs)),
        "ok": float(1.0 if np.all(eigs >= 0.0) else 0.0),
    }
