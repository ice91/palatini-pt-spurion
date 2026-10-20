# palatini_pt/palatini/field_eq.py
# -*- coding: utf-8 -*-
"""
Algebraic field equations (C1): torsion is pure trace and aligns with ∂ε.

最小模型
--------
假設扭率沒有動力學，僅由 spurion 的梯度（外源）代數決定：
    T_μ = α * ∂_μ ε
其中 α 可由理論常數組合而來（本階段以 config/參數傳入）。

輸出
----
- 扭率跡向量 T_μ
- 純跡扭率張量 T_{λμν}（由 T_μ 生成）
- 軸向與純張量分量皆為 0（理論預期 C1）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .connection import minkowski_metric, torsion_from_trace
from .torsion_decomp import decompose


@dataclass
class C1Solution:
    """封裝 C1 的最小解。"""
    T_vec: np.ndarray          # T_μ
    T_tensor: np.ndarray       # T_{λμν}
    g: np.ndarray              # metric
    alpha: float               # 比例係數 α


def solve_torsion_from_spurion(
    d_eps: np.ndarray,
    alpha: float = 1.0,
    g: Optional[np.ndarray] = None,
) -> C1Solution:
    """由 spurion 梯度（∂ε）求代數扭率解（純跡）。"""
    if g is None:
        g = minkowski_metric()
    d_eps = np.asarray(d_eps, dtype=float)
    T_vec = alpha * d_eps
    T_tensor = torsion_from_trace(T_vec, g=g)
    return C1Solution(T_vec=T_vec, T_tensor=T_tensor, g=g, alpha=alpha)


def check_pure_trace(sol: C1Solution, atol: float = 1e-12) -> dict:
    """檢查 axial/pure_tensor 是否 ~ 0。"""
    parts = decompose(sol.T_tensor, sol.g)
    axial_norm = float(np.linalg.norm(parts["axial"]))
    tensor_norm = float(np.linalg.norm(parts["pure_tensor"]))
    ok = (axial_norm < atol) and (tensor_norm < atol)
    return {
        "ok": ok,
        "axial_norm": axial_norm,
        "pure_tensor_norm": tensor_norm,
        "atol": float(atol),
    }


__all__ = ["C1Solution", "solve_torsion_from_spurion", "check_pure_trace"]
