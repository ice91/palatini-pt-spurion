# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict


def order2_raw(alpha: float = 1.0) -> Dict[str, float]:
    r"""
    CS^{++} 建構鏈在 **O(∂²)** 的最小表達。

    與 DBI / closed-metric 鏈等價；這裡故意把 `box_eps` + `d_eps_sq` 之間
    加上一個會被 IBP 視為零的小重新分配（數值上等價），以測試容忍度掃描時的單調收斂性。

    Parameters
    ----------
    alpha : float, default 1.0

    Returns
    -------
    Dict[str, float]
        monomial -> coefficient
    """
    # 以極小數重分配到兩個互通項，等價於加上/減去一個 total derivative：
    eps = 1.0e-16
    return {
        "d_eps_sq": 1.0 + eps,
        "box_eps": -0.5 * alpha - eps,
        "torsion_trace_dot_grad_eps": alpha,
        "torsion_trace_sq": 0.25 * alpha * alpha,
    }
