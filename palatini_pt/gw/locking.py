# -*- coding: utf-8 -*-
"""
palatini_pt.gw.locking
======================

提供「係數鎖定」：解析地施加 L·c = 0，使 c_T^2 ≡ 1。

- `is_locked(coeffs, tol)`：檢查是否滿足 L·c = 0（在公差 tol 內）
- `apply_locking(coeffs, prefer_keys=None)`：調整一個 pivot 係數，令 L·c → 0
- `delta_ct2(coeffs)`：目前的 Δ = L·c
- `lock_and_report(coeffs, tol)`：回傳新係數、Δ 與是否鎖定

鎖定做法：選擇一個「有非零權重 L_i 且當前存在於 coeffs」的 pivot，更新
    c_pivot ← c_pivot - Δ / L_pivot
即可令新的 Δ' = 0。若找不到合適 pivot，會退回到任何有 L_i ≠ 0 的鍵。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from .quadratic_action import locking_weights


def _coef(coeffs: Mapping[str, float], key: str) -> float:
    return float(coeffs.get(key, 0.0))


def delta_ct2(coeffs: Mapping[str, float]) -> float:
    """
    計算 Δ = L·c（c_T^2 = 1 + Δ）。

    Returns
    -------
    float
    """
    L = locking_weights()
    return float(sum(L[k] * _coef(coeffs, k) for k in L))


def is_locked(coeffs: Mapping[str, float], tol: float = 1e-12) -> bool:
    """
    判斷是否已滿足 locking（|Δ| <= tol）。
    """
    return abs(delta_ct2(coeffs)) <= float(tol)


def _pick_pivot(
    coeffs: Mapping[str, float],
    prefer_keys: Optional[Iterable[str]] = None,
) -> Tuple[str, float]:
    """
    根據存在的鍵與非零權重，挑一個可用來解 Δ=0 的 pivot 係數。

    Returns
    -------
    (key, weight)
    """
    L = locking_weights()
    # 先依 prefer_keys 嘗試
    if prefer_keys:
        for k in prefer_keys:
            if k in L and abs(L[k]) > 0.0:
                # 即使 coeffs 中沒有這個鍵，也允許建立它（視為 0 → 可調）
                return k, L[k]
    # 再嘗試 coeffs 已經有的鍵
    for k, w in L.items():
        if abs(w) > 0.0 and (k in coeffs):
            return k, w
    # 最後隨便挑一個 L 中的鍵（建立新鍵）
    for k, w in L.items():
        if abs(w) > 0.0:
            return k, w
    raise RuntimeError("No available pivot to apply locking (all weights are zero?)")


def apply_locking(
    coeffs: Mapping[str, float],
    prefer_keys: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """
    解析地對單一 pivot 係數做調整，令 L·c = 0。

    Parameters
    ----------
    coeffs : Mapping[str, float]
        原始係數（不會就地修改）
    prefer_keys : Iterable[str], optional
        希望優先用來調整的鍵，例如 ["box_eps", "grad_eps2"]。

    Returns
    -------
    dict
        新係數表（已鎖定）
    """
    out: Dict[str, float] = dict(coeffs)  # copy
    Δ = delta_ct2(out)
    if abs(Δ) == 0.0:
        return out  # 已經鎖定

    key, w = _pick_pivot(out, prefer_keys=prefer_keys)
    # c_key ← c_key - Δ / w
    out[key] = float(out.get(key, 0.0) - Δ / w)
    # 驗證鎖定（避免奇異）
    newΔ = delta_ct2(out)
    if abs(newΔ) > 1e-10 * max(1.0, abs(Δ)):
        # 極少見數值不穩時，再做一次微調
        out[key] = float(out[key] - newΔ / w)
    return out


@dataclass
class LockReport:
    before_delta: float
    after_delta: float
    locked: bool
    updated_coeffs: Dict[str, float]


def lock_and_report(
    coeffs: Mapping[str, float],
    tol: float = 1e-12,
    prefer_keys: Optional[Iterable[str]] = None,
) -> LockReport:
    """
    便捷介面：回傳鎖定前/後的 Δ 與新係數表及鎖定判斷。
    """
    before = delta_ct2(coeffs)
    new = apply_locking(coeffs, prefer_keys=prefer_keys)
    after = delta_ct2(new)
    return LockReport(
        before_delta=float(before),
        after_delta=float(after),
        locked=abs(after) <= float(tol),
        updated_coeffs=new,
    )
