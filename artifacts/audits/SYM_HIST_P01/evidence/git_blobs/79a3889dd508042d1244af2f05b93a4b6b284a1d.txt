# tests/test_nlo.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from palatini_pt.gw.nlo import basis_labels_nlo, predict_offsets


def test_nlo_basis_labels_nonempty_and_ascii():
    labels = basis_labels_nlo()
    assert isinstance(labels, list) and len(labels) >= 2
    for s in labels:
        assert isinstance(s, str)
        # sanity: ensure ASCII-only to avoid tooling issues
        s.encode("ascii")


def test_nlo_zero_in_decoupling_limit():
    k = np.logspace(-4, -1, 7)
    cfg = {"nlo": {"gradT_sq_eff": 0.0, "Ricci_deps_deps_eff": 0.0, "Lambda2": 1e12}}
    out = predict_offsets(k, cfg)
    assert np.allclose(out["delta_cT2"], 0.0)
    assert np.allclose(out["delta_K"], 0.0)
    assert np.allclose(out["delta_G"], 0.0)


def test_nlo_scaling_with_k2_over_Lambda2():
    k = np.array([1e-3, 2e-3])
    cfg = {"nlo": {"gradT_sq_eff": 1.0, "Ricci_deps_deps_eff": 0.0, "Lambda2": 1e6}}
    out = predict_offsets(k, cfg)
    ratio = out["delta_cT2"][1] / out["delta_cT2"][0]
    assert np.isclose(ratio, (k[1] / k[0]) ** 2, rtol=1e-12, atol=0)
