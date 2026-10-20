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
    ang = alignment_angle(sol.T_vec, d_eps)
    return C1Report(solution=sol, check=chk, angle_rad=ang)


def smoke_example() -> C1Report:
    """固定一個 ∂ε，給 CI / 測試最小化驗證。"""
    g = minkowski_metric()
    d_eps = np.array([0.3, 0.1, -0.2, 0.4], dtype=float)  # 任意非零梯度
    alpha = 2.5  # 任意比例
    return run_c1(d_eps=d_eps, alpha=alpha, g=g, atol=1e-12)


__all__ = ["alignment_angle", "run_c1", "smoke_example", "C1Report"]
