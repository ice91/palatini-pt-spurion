# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict


def order2_raw(alpha: float = 1.0) -> Dict[str, float]:
    r"""
    DBI 建構鏈在 **O(∂²)** 的最小表達（以 spurion $\epsilon(x)$ 為唯一標記）.

    以最小基底（同 `coeff_extractor.DEFAULT_BASIS`）輸出係數字典：
      - d_eps_sq:           $(\partial\epsilon)^2$
      - box_eps:            $\Box\epsilon$
      - torsion_trace_dot_grad_eps:    $T_\mu\,\partial^\mu\epsilon$
      - torsion_trace_sq:   $T_\mu T^\mu$

    設計為「一致且可對齊」：三條鏈在 O(∂²) 的物理結果應等價，差異僅來自 IBP 可拋之總導數。
    因此這裡先給出一致的解析係數作為最小可用版本。

    Parameters
    ----------
    alpha : float, default 1.0
        一個示意耦合常數，後續可依你的符號推導替換。

    Returns
    -------
    Dict[str, float]
        monomial -> coefficient
    """
    # 一組簡潔、易檢查的解析係數；他鏈將與之 IBP 等價。
    return {
        "d_eps_sq": 1.0,
        "box_eps": -0.5 * alpha,
        "torsion_trace_dot_grad_eps": alpha,
        "torsion_trace_sq": 0.25 * alpha * alpha,
        # 不包含 total_derivative；統一留給其他鏈做 IBP 等價示範。
    }
