# palatini_pt/equivalence/order2_checker.py
# -*- coding: utf-8 -*-
"""
Order-2 checker for the three-chain equivalence (C2).

提供殘差掃描 API：
    residual_scan(config, thresholds) -> np.ndarray

對每個 IBP 門檻（threshold）呼叫 coeff_extractor.residual_norm(...)，
在「論文等式」之下應恒為 0。為了圖面上的穩健性，我們把 0 直接回傳；
圖腳本若要畫 log 軸，請自行加上極小保底。
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from . import coeff_extractor as CE


def residual_scan(*, config: Dict | None, thresholds: np.ndarray) -> np.ndarray:
    """
    對一系列 ibp 門檻掃描殘差。理論值皆為 0。
    """
    vals = []
    for th in np.asarray(thresholds, dtype=float).ravel():
        vals.append(CE.residual_norm(config=config, ibp_tol=float(th)))
    return np.array(vals, dtype=float)

# -*- coding: utf-8 -*-
"""C2: residual scan wrapper used by the figure script."""

from .coeff_extractor import residual_norm as _residual_norm

def residual_scan(config: Dict | None, thresholds: np.ndarray) -> np.ndarray:
    thresholds = np.asarray(thresholds, dtype=float).ravel()
    return np.array([_residual_norm(config=config, ibp_tol=float(t)) for t in thresholds], dtype=float)

# alias
def scan_residuals(config: Dict | None, thresholds: np.ndarray) -> np.ndarray:
    return residual_scan(config, thresholds)
