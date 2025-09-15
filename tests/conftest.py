# tests/conftest.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pytest

@pytest.fixture(scope="session")
def rng():
    seed = int(os.environ.get("PALPT_TEST_SEED", "42"))
    return np.random.default_rng(seed)

@pytest.fixture(scope="session")
def g_minkowski():
    return np.diag([-1.0, 1.0, 1.0, 1.0])

@pytest.fixture(scope="session")
def atol_sym():
    # 與規劃一致：符號層級 ~ 1e-12（此處測的是數值，但我們用同一等級更嚴）
    return 1e-12

@pytest.fixture(scope="session")
def atol_num():
    # 數值層級 ~ 1e-9；某些檢查使用較鬆的數值容忍度
    return 1e-9
