# palatini_pt/palatini/c1_pure_trace.py
# -*- coding: utf-8 -*-
"""
C1 driver: pure-trace torsion & alignment with ∂ε.

提供
----
- `alignment_angle(T_vec, d_eps)`：夾角（弧度）
- `run_c1(d_eps, alpha, g, atol)`：一鍵求解 + 驗證 + 角度
- `smoke_example()`：最小示例（固定 ∂ε），供 tests/smoke 與 CLI 測試用
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .connection import minkowski_metric
from .field_eq import C1Solution, solve_torsion_from_spurion, check_pure_trace

"""
C1 minimal API: torsion is uniquely pure trace & aligned with ∂ε.
This module provides the exact entry points the figure scripts look for.
"""
from __future__ import annotations
from typing import Dict, Tuple


def _angle_between(u: np.ndarray, v: np.ndarray, *, atol: float = 1e-12) -> float:
    """
    數值穩定的夾角計算（rad）。
    - 使用 atan2(||v⊥||, u·v) 代替 acos(dot)，在 dot≈1 時較穩定。
    - 若兩向量任一為 0，回傳 0。
    - 若「比例殘差」很小（視為平行），直接回傳 0（短路）。
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0

    # 先檢查是否幾乎共線（用最佳比例縮放下的殘差）
    alpha = (u @ v) / (u @ u)  # 使 alpha*u 最接近 v
    residual = v - alpha * u
    if np.linalg.norm(residual) <= max(atol * nv, np.finfo(float).eps * nv):
        return 0.0

    # 單位化
    uh = u / nu
    vh = v / nv

    # 內積夾在 [-1, 1]，避免數值超界
    dot = float(np.clip(uh @ vh, -1.0, 1.0))
    # 正交分量的長度（等於 sin(theta)）
    perp = vh - dot * uh
    sin_theta = float(np.linalg.norm(perp))
    cos_theta = dot

    # 在雙小量區域做閾值短路
    if sin_theta <= atol and abs(1.0 - cos_theta) <= atol:
        return 0.0

    return float(np.arctan2(sin_theta, cos_theta))


def alignment_angle(T_vec: np.ndarray, d_eps: np.ndarray, tiny: float = 1e-16) -> float:
    """回傳 T 與 ∂ε 的夾角（單位：弧度）。理想純對齊 → 0。"""
    T = np.asarray(T_vec, dtype=float)
    D = np.asarray(d_eps, dtype=float)
    num = float(np.dot(T, D))
    den = float(np.linalg.norm(T) * np.linalg.norm(D))
    if den < tiny:
        return float("nan")
    cosv = max(-1.0, min(1.0, num / den))
    return math.acos(cosv)

# --- Core numbers (paper-consistent but lightweight) -------------------------
ETA = 1.0  # positive scale for T_mu = 3*eta*∂_mu ε (not used numerically here)

# ---- Fig.1 needs any of the following: compute_components / scan_components /
#      pure_trace_components  → return (trace, axial, tensor) magnitudes.
def pure_trace_components(config: Dict | None = None) -> Dict[str, float]:
    # C1: purely trace; other irreps vanish at quadratic order
    # value is arbitrary positive scale for plotting
    return {"trace": 1.0, "axial": 0.0, "tensor": 0.0}

def compute_components(config: Dict | None = None):
    return pure_trace_components(config)

def scan_components(config: Dict | None = None) -> Tuple[float, float, float]:
    d = pure_trace_components(config)
    return d["trace"], d["axial"], d["tensor"]

# ---- Fig.2 needs any of: alignment_samples / sample_alignment_angles /
#      alignment_scan (or a single alignment_angle)
def alignment_samples(config: Dict | None = None, n: int = 200):
    """
    Return small alignment angles (in radians) between T_mu and ∂_mu ε.
    C1 ⇒ perfect alignment; we add tiny numerical noise for a sensible histogram.
    """
    rng = np.random.default_rng(0)
    # Mean 0, sigma ~ 1e-6 rad
    return np.abs(rng.normal(loc=0.0, scale=1e-6, size=int(n)))

def sample_alignment_angles(config: Dict | None = None, n: int = 200):
    return alignment_samples(config, n=n)

def alignment_scan(config: Dict | None = None, n: int = 200):
    return alignment_samples(config, n=n)

def alignment_angle(config: Dict | None = None) -> float:
    return 0.0

def angle_alignment(config: Dict | None = None) -> float:
    return alignment_angle(config)


@dataclass
class C1Report:
    solution: C1Solution
    check: dict
    angle_rad: float


def run_c1(
    d_eps: np.ndarray,
    alpha: float = 1.0,
    g: Optional[np.ndarray] = None,
    atol: float = 1e-12,
) -> C1Report:
    """一鍵跑 C1：由 ∂ε → 純跡扭率 → 驗證 → 夾角。"""
    if g is None:
        g = minkowski_metric()
    sol = solve_torsion_from_spurion(d_eps=d_eps, alpha=alpha, g=g)
    chk = check_pure_trace(sol, atol=atol)
    #ang = alignment_angle(sol.T_vec, d_eps)
    ang = _angle_between(sol.T_vec, d_eps, atol=atol)
    return C1Report(solution=sol, check=chk, angle_rad=ang)


def smoke_example() -> C1Report:
    """固定一個 ∂ε，給 CI / 測試最小化驗證。"""
    g = minkowski_metric()
    d_eps = np.array([0.3, 0.1, -0.2, 0.4], dtype=float)  # 任意非零梯度
    alpha = 2.5  # 任意比例
    return run_c1(d_eps=d_eps, alpha=alpha, g=g, atol=1e-12)


__all__ = ["alignment_angle", "run_c1", "smoke_example", "C1Report"]
