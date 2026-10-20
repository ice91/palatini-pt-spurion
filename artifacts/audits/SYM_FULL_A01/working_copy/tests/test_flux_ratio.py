# tests/test_flux_ratio.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pytest

from palatini_pt.equivalence.flux_ratio import flux_ratio_FRW


def test_flux_ratio_monotone_closer_to_one():
    Rs = np.array([5.0, 10.0, 20.0, 40.0], dtype=float)
    out = flux_ratio_FRW(Rs, {"flux": {"sigma": 1.0, "c": 0.8}})
    arr = out["R_DBI_CM"]
    # closer to 1 as R grows
    assert arr[-1] < arr[0] and abs(arr[-1] - 1.0) < abs(arr[0] - 1.0)


def test_flux_ratio_requires_positive_R():
    with pytest.raises(ValueError):
        flux_ratio_FRW(np.array([10.0, -1.0]))
