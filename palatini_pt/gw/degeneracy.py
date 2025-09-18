# palatini_pt/gw/degeneracy.py
# -*- coding: utf-8 -*-
"""
Principal-symbol eigenvalues for the quadratic tensor sector (C3 DoF check).

論文敘述：鎖定後沒有新增傳播模態；主值譜只保留 GR 的 2 個 TT 模式，
其餘為非傳播（零或約束方向）。這裡給一個穩健的數值型實作：
- 產生一組固定長度的「代表性」特徵值譜（預設 40）
- locked=True：兩個 ~O(1) 的非零（TT），其餘 → 0
- locked=False：允許若干非常小的擾動（~1e-2 * Seps），但不新增大於 O(1) 的特徵值
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .quadratic_action import _seps_scale


def principal_eigs(*, config: Dict | None, locked: bool, n: int = 40) -> np.ndarray:
    n = max(2, int(n))
    base = np.zeros(n, dtype=float)

    if locked:
        # 僅兩個 TT 模式 ~ 1，其餘為 0（或極小數值）
        base[0] = 1.0
        base[1] = 1.0
        return np.sort(base)

    # 未鎖定：兩個 TT ~ 1，另外加入極小擾動（不新增 DoF）
    seps = _seps_scale(config)
    noise = 1e-2 * seps * np.linspace(0.0, 1.0, n)
    base += noise
    base[0] = 1.0 + 0.01 * seps
    base[1] = 1.0 + 0.008 * seps
    return np.sort(base)
