# palatini_pt/equivalence/order2_checker.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict
import numpy as np
from . import coeff_extractor as CE

def residual_scan(*, config: Dict | None, thresholds: np.ndarray) -> np.ndarray:
    ts = np.asarray(thresholds, dtype=float).ravel()
    vals = [float(CE.residual_norm(config=config, ibp_tol=float(t))) for t in ts]
    return np.asarray(vals, dtype=float)

# alias
def scan_residuals(config: Dict | None, thresholds: np.ndarray) -> np.ndarray:
    return residual_scan(config=config, thresholds=thresholds)
