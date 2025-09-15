# -*- coding: utf-8 -*-
"""
palatini_pt.gw.tensor_mode
==========================

基於二次作用量參數 (Z, c_T^2) 的張量模式工具：

- `cT_of_k(k, coeffs, lock=False)`：回傳 c_T(k)；若 lock=True，先套用 locking。
- `dispersion_omega2(k, coeffs, lock=False)`：回傳 ω^2 = c_T^2 k^2。
- `sample_cT_grid(k_list, coeffs_list, lock=False)`：把多組係數與 k 串起來產生表格。
"""
from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .locking import apply_locking
from .quadratic_action import quadratic_action_params


def _ensure_1d(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def cT_of_k(k: Iterable[float], coeffs: Mapping[str, float], lock: bool = False) -> np.ndarray:
    """
    計算 c_T(k)。此最小模型中 c_T 與 k 無關；若 lock=True，將會 c_T ≡ 1。

    Parameters
    ----------
    k : Iterable[float]
        波數列表
    coeffs : Mapping[str, float]
        係數表
    lock : bool, default False
        是否先套用 locking（令 L·c = 0）

    Returns
    -------
    np.ndarray
        與 k 等長的一維陣列（全部相同的 c_T）
    """
    if lock:
        coeffs = apply_locking(coeffs)
    pars = quadratic_action_params(coeffs)
    cT2 = float(pars["cT2"])
    # 若 cT2 因浮點誤差微負，做最小截斷
    cT = float(np.sqrt(max(cT2, 0.0)))
    k_arr = _ensure_1d(k)
    return np.full_like(k_arr, fill_value=cT, dtype=float)


def dispersion_omega2(k: Iterable[float], coeffs: Mapping[str, float], lock: bool = False) -> np.ndarray:
    """
    ω^2 = c_T^2 k^2。lock=True 時，c_T^2 = 1 → ω^2 = k^2。
    """
    if lock:
        coeffs = apply_locking(coeffs)
    pars = quadratic_action_params(coeffs)
    cT2 = float(pars["cT2"])
    k_arr = _ensure_1d(k)
    return cT2 * k_arr * k_arr


def sample_cT_grid(
    k_list: Iterable[float],
    coeffs_list: Sequence[Mapping[str, float]],
    lock: bool = False,
) -> np.ndarray:
    """
    產生 |coeffs| x |k| 的 c_T 網格（行：k，列：不同係數組）。

    Returns
    -------
    np.ndarray
        形狀 (len(k), len(coeffs_list))
    """
    ks = _ensure_1d(k_list)
    out = np.zeros((ks.size, len(coeffs_list)), dtype=float)
    for j, c in enumerate(coeffs_list):
        out[:, j] = cT_of_k(ks, c, lock=lock)
    return out
