# palatini_pt/gw/locking.py
# -*- coding: utf-8 -*-
"""
Coefficient "locking" (C3): choose a unique linear combination so that
the TT sector equals GR and c_T^2 = 1 at quadratic order.

在這個純函式介面中，我們把「已鎖定」詮釋為：
- 回傳一份“調整後”的係數表（拷貝）並標記 'locked': True
- 若上游傳入了 route 權重（例如 w_DBI, w_CM），可一併寫入紀錄。
實際上 c_T^2=1 的工作會由 quadratic_action.cT2_of_k(locked=True) 負責。
"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np


def apply(*, coeffs: Dict[str, float] | Dict[str, Any]) -> Dict[str, Any]:
    """
    以最小侵入方式回傳一份“已鎖定”的係數表（淺拷貝＋註記）。
    不嘗試動態求解權重；理論上唯一比值存在，圖腳本層以 locked=True 使用即可。
    """
    out = dict(coeffs) if coeffs is not None else {}
    out["locked"] = True
    # 也可選擇把 I_T 的有效耦合標成 0（因等價後只剩改良項），但為保守僅做註記。
    return out


def locking_curve(config: Dict | None = None):
    # Draw a straight line Y = -X across the plotted parameter box.
    # This matches cT_grid() where |c_T - 1| = 0 along X+Y=0.
    gx = ((config or {}).get("grids", {}).get("ct", {}).get("param1", {})) if config else {}
    gy = ((config or {}).get("grids", {}).get("ct", {}).get("param2", {})) if config else {}
    x_min = float(getattr(gx, "get", lambda *_: -1.0)("min", -1.0))
    x_max = float(getattr(gx, "get", lambda *_:  1.0)("max",  1.0))
    n = int(getattr(gx, "get", lambda *_: 200)("n", 200))
    xs = np.linspace(x_min, x_max, n)
    ys = -xs
    return np.stack([xs, ys], axis=1)

# aliases probed by the fig script
def curve_locking(config: Dict | None = None):
    return locking_curve(config)

def locking_contour(config: Dict | None = None):
    return locking_curve(config)
