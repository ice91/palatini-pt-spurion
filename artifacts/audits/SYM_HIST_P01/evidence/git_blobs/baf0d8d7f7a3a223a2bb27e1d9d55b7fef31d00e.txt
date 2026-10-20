# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict


def order2_raw(alpha: float = 1.0) -> Dict[str, float]:
    r"""
    Closed-metric 建構鏈在 **O(∂²)** 的最小表達。

    與 DBI 鏈等價，但刻意加入一項 **IBP 可丟的總導數** 來模擬實際「規則重寫」情境：
    - total_derivative:    $\nabla_\mu(\epsilon\,T^\mu)$ 之類（此鍵會在 `coeff_extractor` 被忽略）

    Parameters
    ----------
    alpha : float, default 1.0

    Returns
    -------
    Dict[str, float]
        monomial -> coefficient
    """
    return {
        "d_eps_sq": 1.0,
        "box_eps": -0.5 * alpha,
        "torsion_trace_dot_grad_eps": alpha,
        "torsion_trace_sq": 0.25 * alpha * alpha,
        # 代表「可被 IBP 丟棄」的鍵，extractor 會忽略，不進入共同基底：
        "total_derivative": 3.1415926535e-3,  # 任意小常數（不進入基底，僅示範）
    }
