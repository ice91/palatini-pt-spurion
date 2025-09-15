# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np
import pytest

from palatini_pt.equivalence import (
    compute_equivalence_report,
    get_basis_labels,
    coeff_vector_named,
    dbi_order2,
    closed_metric_order2,
    cspp_order2,
)


# ---- 小工具：把 residual 在 |x|<tol 的元素視為 0，再計算 L2 ----
def _norm_with_tol(x: np.ndarray, tol: float) -> float:
    masked = np.where(np.abs(x) < tol, 0.0, x)
    return float(np.linalg.norm(masked))


def _is_monotone_nonincreasing(xs: Iterable[float]) -> bool:
    xs = list(xs)
    return all(xs[i] >= xs[i + 1] - 1e-18 for i in range(len(xs) - 1))


@pytest.mark.parametrize("alpha", [-3.0, -0.7, 0.0, 1e-6, 1.0, 4.2])
def test_order2_pairwise_residuals_close_to_zero(alpha: float):
    """
    核心等價性：三鏈在 O(∂²) 兩兩殘差 ~ 0。
    門檻參考專案規劃：符號級 1e-12、數值級 1e-9。
    """
    rep = compute_equivalence_report(alpha=alpha)
    # 共同基底存在且非空
    assert isinstance(rep.basis, list) and len(rep.basis) > 0

    # 兩兩殘差 L2
    n_d_c = rep.norms["dbi-closed"]
    n_d_s = rep.norms["dbi-cspp"]
    n_c_s = rep.norms["closed-cspp"]

    # 這套最小實作的三鏈係數是解析匹配的（IBP鍵已忽略），理想狀況直接 ~0
    # 若未來你把三鏈換成真實推導式，允許落在 1e-12（符號）～1e-9（數值）之內
    assert n_d_c < 1e-12 or n_d_c < 1e-9
    assert n_d_s < 1e-12 or n_d_s < 1e-9
    assert n_c_s < 1e-12 or n_c_s < 1e-9


def test_ibp_ignored_key_is_effectively_dropped():
    """
    closed-metric 鏈有 'total_derivative'，抽取係數向量後與 DBI 仍一致。
    """
    labels = get_basis_labels()
    raw_closed = closed_metric_order2(alpha=1.0)
    raw_dbi = dbi_order2(alpha=1.0)

    # 原始 closed 內確實有 IBP 可丟鍵
    assert "total_derivative" in raw_closed

    v_closed = coeff_vector_named(raw_closed, labels).vec
    v_dbi = coeff_vector_named(raw_dbi, labels).vec

    # 對齊後相等（或在浮點極小誤差內）
    diff = np.linalg.norm(v_closed - v_dbi)
    assert diff < 1e-15


def test_tolerance_sweep_is_monotone_nonincreasing():
    """
    對殘差做「容忍度掃描」：把 |delta|<tol 視為 0，L2 應單調不增。
    用 CS++ 鏈裡的極小重分配（~1e-16）作為掃描的可感知量級。
    """
    rep = compute_equivalence_report(alpha=1.0)
    res = rep.residuals["dbi-cspp"]

    # 掃一串容忍度：由小到大
    tols = [0.0, 1e-18, 1e-16, 1e-14, 1e-12, 1e-9]
    norms = [_norm_with_tol(res, tol) for tol in tols]

    assert _is_monotone_nonincreasing(norms), f"norms not monotone: {norms}"


def test_override_basis_order_does_not_change_equivalence():
    """
    自訂基底順序（打亂預設順序）後，等價性仍成立。
    """
    default = get_basis_labels()
    assert len(default) >= 4  # 最小基底大小

    # 打亂順序（這裡用反轉，保證可重現）
    custom = list(reversed(default))

    rep_custom = compute_equivalence_report(alpha=2.5, basis_labels=custom)
    # 檢查長度與標籤完全照我們指定
    assert rep_custom.basis == custom
    # 檢查兩兩殘差仍 ~ 0
    for k, v in rep_custom.norms.items():
        assert v < 1e-12 or v < 1e-9


def test_as_table_rows_match_basis():
    """
    as_table() 回傳列數與基底一致，且可轉 float。
    """
    rep = compute_equivalence_report(alpha=0.3)
    rows = rep.as_table()
    assert len(rows) == len(rep.basis)
    # 每列是 (name, c_dbi, c_closed, c_cspp)
    for name, c1, c2, c3 in rows:
        assert isinstance(name, str)
        # 能被轉 float（避免奇怪型別）
        _ = float(c1)
        _ = float(c2)
        _ = float(c3)
