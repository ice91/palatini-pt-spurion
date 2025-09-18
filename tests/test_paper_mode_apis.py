# tests/test_paper_mode_apis.py
# -*- coding: utf-8 -*-
import importlib
import numpy as np


def test_equivalence_real_api_present():
    oc = importlib.import_module("palatini_pt.equivalence.order2_checker")
    ce = importlib.import_module("palatini_pt.equivalence.coeff_extractor")
    assert hasattr(oc, "residual_scan")
    assert hasattr(ce, "coeff_vectors")
    assert hasattr(ce, "residual_norm")


def test_gw_real_api_present_and_locked_ct():
    tm = importlib.import_module("palatini_pt.gw.tensor_mode")
    qa = importlib.import_module("palatini_pt.gw.quadratic_action")
    lock = importlib.import_module("palatini_pt.gw.locking")
    deg = importlib.import_module("palatini_pt.gw.degeneracy")

    # cT 存在且 locked 後恆為 1
    k = np.logspace(-4, -1, 8)
    cT2_locked = qa.cT2_of_k(k=k, config={"spurion": {"seps_scale": 0.3}}, locked=True)
    cT_locked = tm.cT_of_k(k=k, config={"spurion": {"seps_scale": 0.3}}, locked=True)
    assert np.allclose(cT2_locked, 1.0)
    assert np.allclose(cT_locked, 1.0)

    # locking.apply() 介面存在
    out = lock.apply(coeffs={"I_T": 1.0})
    assert out.get("locked", False) is True

    # 退化性：鎖定後只有兩個 O(1) 的特徵值
    eigs = deg.principal_eigs(config={"spurion": {"seps_scale": 0.3}}, locked=True, n=40)
    assert (eigs >= -1e-12).all()
    assert (eigs[-1] > 0.9) and (eigs[-2] > 0.9)  # 兩個 TT
