# -*- coding: utf-8 -*-
import math
import copy
import numpy as np
import pytest

from palatini_pt.gw.quadratic_action import (
    get_ct2_from_coeffs,
    get_Z_from_coeffs,
    quadratic_action_params,
    locking_weights,
)
from palatini_pt.gw.locking import (
    is_locked,
    apply_locking,
    lock_and_report,
    delta_ct2,
)
from palatini_pt.gw.tensor_mode import (
    cT_of_k,
    dispersion_omega2,
    sample_cT_grid,
)
from palatini_pt.gw.degeneracy import (
    principal_symbol_eigs,
    is_nondegenerate,
    spectrum_report,
)

# 全域數值公差（對應規劃中的 Gate C）
LOCK_TOL = 1e-10
NUM_TOL = 1e-12


def _delta_manual(coeffs):
    """用目前 gw/quadratic_action.py 的 L 權重重算 Δ = L·c（以防權重更新時測試一起對齊）"""
    L = locking_weights()
    return sum(L[k] * float(coeffs.get(k, 0.0)) for k in L)


@pytest.fixture
def k_grid():
    return np.array([1e-4, 1e-3, 1e-2, 1e-1], dtype=float)


@pytest.fixture
def coeffs_pos():
    # 給一組會讓 c_T^2 > 1 的係數
    # Δ = 0.25*0.02 + (-0.5)*(-0.01) + 0.125*0.005 = 0.010625
    # cT2 = 1.010625 → cT ≈ 1.00529
    return {"grad_eps2": 0.02, "box_eps": -0.01, "torsion_trace_grad_eps": 0.005}


@pytest.fixture
def coeffs_neg_ct2():
    # 給一組讓 c_T^2 < 0（未鎖定會顯示病態；鎖定後恢復 c_T=1）
    # 取 box_eps = +3 → Δ = -0.5*3 = -1.5 → cT2 = -0.5
    return {"grad_eps2": 0.0, "box_eps": 3.0, "torsion_trace_grad_eps": 0.0}


def test_ct2_and_Z_match_analytic(coeffs_pos):
    # 驗證 get_ct2_from_coeffs / get_Z_from_coeffs 的解析式與手算一致
    ct2 = get_ct2_from_coeffs(coeffs_pos)
    Z = get_Z_from_coeffs(coeffs_pos)

    # 手算 Δ 與 Z
    delta = _delta_manual(coeffs_pos)
    ge2 = abs(float(coeffs_pos.get("grad_eps2", 0.0)))
    bxe = abs(float(coeffs_pos.get("box_eps", 0.0)))
    Z_expected = 1.0 + 0.10 * ge2 + 0.05 * bxe

    assert math.isclose(ct2, 1.0 + delta, rel_tol=0, abs_tol=NUM_TOL)
    assert math.isclose(Z, Z_expected, rel_tol=0, abs_tol=NUM_TOL)


def test_locking_makes_cT_equal_one_everywhere(k_grid, coeffs_pos):
    # 未鎖定：c_T(k) 偏離 1
    cT_unlocked = cT_of_k(k_grid, coeffs_pos, lock=False)
    assert np.max(np.abs(cT_unlocked - 1.0)) > 1e-6

    # 鎖定：c_T(k) ≡ 1
    cT_locked = cT_of_k(k_grid, coeffs_pos, lock=True)
    assert np.max(np.abs(cT_locked - 1.0)) < LOCK_TOL


def test_locking_delta_zero_and_idempotent(coeffs_pos):
    # Δ 前後 → 0
    d0 = delta_ct2(coeffs_pos)
    rep = lock_and_report(coeffs_pos, tol=LOCK_TOL)
    d1 = rep.after_delta
    assert abs(d0) > 0.0
    assert abs(d1) <= LOCK_TOL
    assert is_locked(rep.updated_coeffs, tol=LOCK_TOL)

    # 再鎖一次不應改變（idempotent within tolerance）
    coeffs2 = apply_locking(rep.updated_coeffs)
    for k in set(coeffs2) | set(rep.updated_coeffs):
        assert math.isclose(
            float(coeffs2.get(k, 0.0)),
            float(rep.updated_coeffs.get(k, 0.0)),
            rel_tol=0,
            abs_tol=1e-12,
        )


def test_dispersion_before_after_lock(k_grid, coeffs_pos):
    # 未鎖定：ω^2 = cT2 k^2
    pars = quadratic_action_params(coeffs_pos)
    cT2 = pars["cT2"]
    w2_unlocked = dispersion_omega2(k_grid, coeffs_pos, lock=False)
    assert np.allclose(w2_unlocked, cT2 * k_grid * k_grid, atol=NUM_TOL, rtol=0)

    # 鎖定：ω^2 = k^2
    w2_locked = dispersion_omega2(k_grid, coeffs_pos, lock=True)
    assert np.allclose(w2_locked, k_grid * k_grid, atol=LOCK_TOL, rtol=0)


def test_principal_symbol_degeneracy_flow(coeffs_neg_ct2):
    # 未鎖定：cT2 < 0 → 主值一正一負 → nondegenerate=False
    eigs0 = principal_symbol_eigs(coeffs_neg_ct2)
    assert eigs0.shape == (2,)
    assert not is_nondegenerate(coeffs_neg_ct2)

    # 鎖定：cT2 → 1 → 兩特徵值非負
    fixed = apply_locking(coeffs_neg_ct2)
    assert is_nondegenerate(fixed)
    eigs1 = principal_symbol_eigs(fixed)
    assert np.all(eigs1 >= -1e-14)

    rep = spectrum_report(fixed)
    assert rep["cT"] == pytest.approx(1.0, abs=LOCK_TOL)
    assert rep["Z"] > 0.0
    assert rep["eig_min"] >= -1e-14


def test_sample_cT_grid_shape_and_values(k_grid, coeffs_pos):
    c_other = {"grad_eps2": -0.01, "box_eps": 0.0, "torsion_trace_grad_eps": 0.0}
    grid = sample_cT_grid(k_grid, [coeffs_pos, c_other], lock=False)
    assert grid.shape == (k_grid.size, 2)

    # 每一欄都應該是常數（本模型 c_T 與 k 無關）
    assert np.allclose(grid[:, 0], grid[0, 0], atol=NUM_TOL, rtol=0)
    assert np.allclose(grid[:, 1], grid[0, 1], atol=NUM_TOL, rtol=0)

    # 鎖定後整個網格應該都是 1
    grid_locked = sample_cT_grid(k_grid, [coeffs_pos, c_other], lock=True)
    assert np.allclose(grid_locked, 1.0, atol=LOCK_TOL, rtol=0)


def test_locking_respects_prefer_keys(coeffs_pos):
    # 指定 prefer_keys，檢查只有該鍵被修改（其餘鍵保持）
    before = copy.deepcopy(coeffs_pos)
    after = apply_locking(before, prefer_keys=("box_eps",))

    # Δ 必須被消掉
    assert abs(delta_ct2(after)) <= LOCK_TOL

    # 只有 box_eps 應該有變動（其餘鍵與 before 相同）
    for k in before:
        if k == "box_eps":
            continue
        assert math.isclose(
            float(before[k]), float(after.get(k, 0.0)), rel_tol=0, abs_tol=1e-12
        )
