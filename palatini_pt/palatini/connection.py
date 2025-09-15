# palatini_pt/palatini/connection.py
# -*- coding: utf-8 -*-
"""
Connection and contorsion helpers (minimal, algebraic).

目標
----
本模組提供在「純跡扭率」假設下，連接與扭率/擬曲率的最小建構：
- 以 metric g_{μν} 與「扭率跡向量」T_μ 生成對應的扭率張量
  T_{λμν} = (1/3)(g_{λν} T_μ - g_{λμ} T_ν)
- 對應的 contorsion（純跡情形）
  K_{λμν} = (1/3)(g_{λμ} T_ν - g_{λν} T_μ)

這些式子對應到經典分解的純跡分量；當你需要升級到
一般連接 Γ = { } + K 時，只要把 Levi-Civita({}) 接成數值或符號即可。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def minkowski_metric(sign: int = -1) -> np.ndarray:
    """回傳 4D Minkowski metric（diag(sign, +1, +1, +1)）。

    Parameters
    ----------
    sign : int
        時間分量的符號，預設 -1（mostly plus）。
    """
    return np.diag([sign, 1.0, 1.0, 1.0])


def torsion_from_trace(T_vec: np.ndarray, g: Optional[np.ndarray] = None) -> np.ndarray:
    """由「扭率跡向量」建立純跡扭率張量 T_{λμν}。

    定義
    ----
    T_{λμν} = (1/3)(g_{λν} T_μ - g_{λμ} T_ν)

    Notes
    -----
    - 回傳的 T 具有反稱性 T_{λμν} = - T_{λνμ}
    - 維度預設 4；若 g 提供其他維度亦可，但公式仍採上述純跡型式。
    """
    if g is None:
        g = minkowski_metric()
    g = np.asarray(g, dtype=float)
    T_vec = np.asarray(T_vec, dtype=float)
    dim = g.shape[0]
    assert T_vec.shape == (dim,)

    T = np.zeros((dim, dim, dim), dtype=float)
    for lam in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                T[lam, mu, nu] = (1.0 / 3.0) * (g[lam, nu] * T_vec[mu] - g[lam, mu] * T_vec[nu])
    return T


def contorsion_from_trace(T_vec: np.ndarray, g: Optional[np.ndarray] = None) -> np.ndarray:
    """純跡扭率對應的 contorsion K_{λμν}。

    K_{λμν} = (1/3)(g_{λμ} T_ν - g_{λν} T_μ)
    """
    if g is None:
        g = minkowski_metric()
    g = np.asarray(g, dtype=float)
    T_vec = np.asarray(T_vec, dtype=float)
    dim = g.shape[0]
    assert T_vec.shape == (dim,)

    K = np.zeros((dim, dim, dim), dtype=float)
    for lam in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                K[lam, mu, nu] = (1.0 / 3.0) * (g[lam, mu] * T_vec[nu] - g[lam, nu] * T_vec[mu])
    return K


__all__ = ["minkowski_metric", "torsion_from_trace", "contorsion_from_trace"]
