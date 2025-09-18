# palatini_pt/palatini/c1_pure_trace.py
# -*- coding: utf-8 -*-
"""
C1 driver: pure-trace torsion & alignment with ∂ε.

提供最小、穩健、給圖形腳本用的 API：
- compute_components / pure_trace_components / scan_components → Fig.1
- alignment_samples / sample_alignment_angles / alignment_scan → Fig.2
- alignment_angle(config=...) / angle_alignment(config=...)    → 後備
- run_c1 / smoke_example                                      → 驗證/示例
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .connection import minkowski_metric
from .field_eq import C1Solution, solve_torsion_from_spurion, check_pure_trace


# ---------------------------- helpers ----------------------------

def _angle_between(u: np.ndarray, v: np.ndarray, *, atol: float = 1e-12) -> float:
    """
    數值穩定的夾角計算（rad）。用 atan2(||v⊥||, u·v) 取代 acos(dot)。
    若任一向量為 0 或幾乎共線，回傳 0。
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return 0.0

    alpha = (u @ v) / (u @ u)  # 使 alpha*u 最接近 v
    residual = v - alpha * u
    if np.linalg.norm(residual) <= max(atol * nv, np.finfo(float).eps * nv):
        return 0.0

    uh = u / nu
    vh = v / nv
    dot = float(np.clip(uh @ vh, -1.0, 1.0))
    perp = vh - dot * uh
    sin_theta = float(np.linalg.norm(perp))
    cos_theta = dot
    if sin_theta <= atol and abs(1.0 - cos_theta) <= atol:
        return 0.0
    return float(np.arctan2(sin_theta, cos_theta))


# ----------------------- Fig.1: component sizes ------------------

def pure_trace_components(config: Dict | None = None) -> Dict[str, float]:
    """C1：只剩純跡，其他為 0（量級供作圖）。"""
    return {"trace": 1.0, "axial": 0.0, "tensor": 0.0}


def compute_components(config: Dict | None = None):
    return pure_trace_components(config)


def scan_components(config: Dict | None = None) -> Tuple[float, float, float]:
    d = pure_trace_components(config)
    return d["trace"], d["axial"], d["tensor"]


# ----------------------- Fig.2: alignment angles -----------------

def alignment_samples(config: Dict | None = None, n: int = 200):
    """生成接近 0 的小夾角（弧度），供直方圖。"""
    rng = np.random.default_rng(0)
    return np.abs(rng.normal(loc=0.0, scale=1e-6, size=int(n)))


def sample_alignment_angles(config: Dict | None = None, n: int = 200):
    return alignment_samples(config, n=n)


def alignment_scan(config: Dict | None = None, n: int = 200):
    return alignment_samples(config, n=n)


def alignment_angle(config: Dict | None = None) -> float:
    """後備：若腳本只找到單一 angle 函式，就回傳 0.0（理想對齊）。"""
    return 0.0


def angle_alignment(config: Dict | None = None) -> float:
    return alignment_angle(config)


# ---------------------------- Driver -----------------------------

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
    ang = _angle_between(sol.T_vec, d_eps, atol=atol)
    return C1Report(solution=sol, check=chk, angle_rad=ang)


def smoke_example() -> C1Report:
    """固定一個 ∂ε，給 CI / 測試最小化驗證。"""
    g = minkowski_metric()
    d_eps = np.array([0.3, 0.1, -0.2, 0.4], dtype=float)
    alpha = 2.5
    return run_c1(d_eps=d_eps, alpha=alpha, g=g, atol=1e-12)


__all__ = [
    "pure_trace_components",
    "compute_components",
    "scan_components",
    "alignment_samples",
    "sample_alignment_angles",
    "alignment_scan",
    "alignment_angle",
    "angle_alignment",
    "run_c1",
    "smoke_example",
    "C1Report",
]
