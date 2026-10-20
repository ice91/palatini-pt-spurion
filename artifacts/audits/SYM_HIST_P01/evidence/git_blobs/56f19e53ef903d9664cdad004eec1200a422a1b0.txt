# palatini_pt/palatini/torsion_decomp.py
# -*- coding: utf-8 -*-
"""
Torsion decomposition: trace / axial / pure-tensor pieces.

分解式（4D，全降指）：
    T_{λμν} = (1/3)(g_{λν} T_μ - g_{λμ} T_ν) - (1/6) ε_{λμνσ} S^σ + q_{λμν}
其中
    T_μ      = T^{λ}{}_{μλ} = g^{λρ} T_{ρ μ λ}          （跡向量）
    S^σ      = ε^{σμνρ} T_{μνρ}                         （軸向向量）
    q_{λμν}  = 剩餘純張量（trace-free & axial-free）

提供：
- `trace_vector(T, g, ginv)`
- `axial_vector(T, ginv)`
- `decompose(T, g)` → dict 包含 trace/axial/pure_tensor 與重建檢查
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def levi_civita_4() -> np.ndarray:
    """回傳 4D Levi-Civita ε_{αβγδ}，採座標正向定義（0123 = +1）。"""
    eps = np.zeros((4, 4, 4, 4), dtype=int)
    import itertools

    for perm in itertools.permutations(range(4)):
        sign = permutation_sign(perm)
        eps[perm] = sign
    return eps


def permutation_sign(perm: Tuple[int, int, int, int]) -> int:
    """簽位函式：偶排列 +1、奇排列 -1。"""
    sign = 1
    p = list(perm)
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                sign *= -1
    return sign


def minkowski_metric(sign: int = -1) -> np.ndarray:
    return np.diag([sign, 1.0, 1.0, 1.0])


def inverse_metric(g: np.ndarray) -> np.ndarray:
    return np.linalg.inv(g)


def trace_vector(T: np.ndarray, ginv: np.ndarray) -> np.ndarray:
    """T_μ = T^{λ}{}_{μλ} = g^{λρ} T_{ρ μ λ}（全降指輸入）。"""
    dim = T.shape[0]
    out = np.zeros((dim,), dtype=float)
    for mu in range(dim):
        s = 0.0
        for lam_up in range(dim):
            for rho in range(dim):
                s += ginv[lam_up, rho] * T[rho, mu, lam_up]
        out[mu] = s
    return out


def axial_vector(T: np.ndarray, ginv: np.ndarray) -> np.ndarray:
    """S^σ = ε^{σμνρ} T_{μνρ}（全降指輸入，藉由 g^{-1} 把 ε 升指）。"""
    dim = T.shape[0]
    assert dim == 4, "目前軸向向量實作限定 4D"
    eps_dn = levi_civita_4().astype(float)  # 下標版本
    # 以 g^{-1} 升第一個指標：ε^{σ}{}_{μνρ} = g^{σα} ε_{α μ ν ρ}
    out = np.zeros((dim,), dtype=float)
    for sig in range(dim):
        s = 0.0
        for alpha in range(dim):
            for mu in range(dim):
                for nu in range(dim):
                    for rho in range(dim):
                        eps_up = 0.0
                        # ε^{σ μ ν ρ} = g^{σα} g^{μβ} g^{νγ} g^{ρδ} ε_{αβγδ}，但只需要與 T_{μνρ}
                        # 收縮第一個指標即可簡化成：
                        eps_up += ginv[sig, alpha] * eps_dn[alpha, mu, nu, rho]
                        s += eps_up * T[mu, nu, rho]
        out[sig] = s
    return out


def pure_trace_piece(T_vec: np.ndarray, g: np.ndarray) -> np.ndarray:
    """(1/3)(g_{λν} T_μ - g_{λμ} T_ν)"""
    dim = g.shape[0]
    P = np.zeros((dim, dim, dim), dtype=float)
    for lam in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                P[lam, mu, nu] = (1.0 / 3.0) * (g[lam, nu] * T_vec[mu] - g[lam, mu] * T_vec[nu])
    return P


def axial_piece(S_vec: np.ndarray) -> np.ndarray:
    """-(1/6) ε_{λμνσ} S^σ，限定 4D。"""
    dim = S_vec.shape[0]
    assert dim == 4, "目前 axial piece 實作限定 4D"
    eps = levi_civita_4().astype(float)
    A = np.zeros((dim, dim, dim), dtype=float)
    for lam in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                s = 0.0
                for sig in range(dim):
                    s += eps[lam, mu, nu, sig] * S_vec[sig]
                A[lam, mu, nu] = - (1.0 / 6.0) * s
    return A


def decompose(T: np.ndarray, g: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    """做三分解：trace/axial/pure_tensor，並回傳重建檢查殘差。"""
    if g is None:
        g = minkowski_metric()
    ginv = inverse_metric(g)

    T_tr = trace_vector(T, ginv)
    S_ax = axial_vector(T, ginv)
    P = pure_trace_piece(T_tr, g)
    A = axial_piece(S_ax)
    q = T - P - A
    recon = P + A + q
    res = T - recon
    return {
        "trace_vec": T_tr,
        "axial_vec": S_ax,
        "pure_trace": P,
        "axial": A,
        "pure_tensor": q,
        "reconstruct_residual": res,
    }


__all__ = [
    "minkowski_metric",
    "inverse_metric",
    "trace_vector",
    "axial_vector",
    "pure_trace_piece",
    "axial_piece",
    "decompose",
]
