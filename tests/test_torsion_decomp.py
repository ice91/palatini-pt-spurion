# tests/test_torsion_decomp.py
# -*- coding: utf-8 -*-
import numpy as np
import pytest

from palatini_pt.palatini import (
    minkowski_metric,
    torsion_from_trace,
    decompose,
)

def test_decomposition_of_pure_trace(g_minkowski):
    g = g_minkowski
    T_vec = np.array([0.7, -0.2, 0.5, 0.9], dtype=float)
    T = torsion_from_trace(T_vec, g=g)

    parts = decompose(T, g)
    # 抓回的 trace_vec 應與原本一致（數值上允許極小誤差）
    assert np.allclose(parts["trace_vec"], T_vec, atol=1e-14, rtol=0.0)

    # axial/pure_tensor ~ 0；重建殘差 ~ 0
    assert np.linalg.norm(parts["axial"]) < 1e-14
    assert np.linalg.norm(parts["pure_tensor"]) < 1e-14
    assert np.linalg.norm(parts["reconstruct_residual"]) < 1e-14
