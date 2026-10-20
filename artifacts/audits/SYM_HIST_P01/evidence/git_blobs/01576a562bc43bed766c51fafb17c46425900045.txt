# tests/test_c1_torsion.py
# -*- coding: utf-8 -*-
import math
import numpy as np
import pytest

from palatini_pt.palatini import (
    run_c1,
    decompose,
    minkowski_metric,
)

@pytest.mark.parametrize(
    "d_eps",
    [
        np.array([0.3, 0.1, -0.2, 0.4], dtype=float),  # 固定向量（與 docs 一致）
        np.array([1.0, 0.0, 0.0, 0.0], dtype=float),   # time-like
        np.array([0.0, 1.0, 0.0, 0.0], dtype=float),   # x-like
        np.array([0.2, 0.2, 0.2, 0.2], dtype=float),   # 均勻
    ],
)
def test_c1_pure_trace_and_alignment(d_eps, g_minkowski, atol_sym):
    rep = run_c1(d_eps=d_eps, alpha=2.5, g=g_minkowski, atol=atol_sym)

    # 1) 純跡檢查（非純跡分量的 Frobenius norm 近 0）
    assert rep.check["ok"], f"axial={rep.check['axial_norm']} tensor={rep.check['pure_tensor_norm']}"

    # 2) 三分解重建
    parts = decompose(rep.solution.T_tensor, rep.solution.g)
    recon_res = parts["reconstruct_residual"]
    assert np.linalg.norm(recon_res) < atol_sym

    # 3) 對齊角度（rad）應接近 0
    assert rep.angle_rad == rep.angle_rad  # 非 NaN
    assert abs(rep.angle_rad) < 1e-14  # 更嚴格（數學上應該精確 0）


def test_c1_random_many(rng, g_minkowski, atol_sym):
    # 多組隨機 ∂ε 驗證
    for _ in range(20):
        d_eps = rng.normal(size=4)
        # 避免零向量（角度會是 NaN）
        if np.linalg.norm(d_eps) < 1e-15:
            d_eps[0] = 1.0
        rep = run_c1(d_eps=d_eps, alpha=rng.uniform(0.1, 3.0), g=g_minkowski, atol=atol_sym)
        assert rep.check["ok"]
        assert abs(rep.angle_rad) < 1e-12
