# palatini_pt/equivalence/flux_ratio.py
# -*- coding: utf-8 -*-
"""
Flux-ratio invariant (C2) — FRW finite-ball convergence toy.

For a comoving sphere of radius R, the ratio between improvement-current fluxes
from two bulk-equivalent routes approaches 1 as R → ∞ with a power-law tail
set by boundary fall-offs. This module provides a minimal, reproducible toy
for plotting and tests.
"""
from __future__ import annotations

from typing import Dict
import numpy as np


def flux_ratio_FRW(R_vals: np.ndarray, config: Dict | None = None) -> Dict[str, np.ndarray]:
    """
    Return a toy convergence profile of the DBI/CM flux ratio on FRW finite balls.

    Model:
        R_{DBI/CM}(R) = 1 + c * R^{-sigma}

    Parameters
    ----------
    R_vals : array_like
        Radii to evaluate the ratio on (must be positive).
    config : dict or None
        Optional settings under key 'flux':
            {"flux": {"sigma": <float>,   # default 1.0
                      "c": <float>}}      # default 0.5

    Returns
    -------
    dict with:
        'R'          : np.ndarray of radii
        'R_DBI_CM'   : np.ndarray of ratios, → 1 as R grows
    """
    R_vals = np.asarray(R_vals, dtype=float)
    if np.any(R_vals <= 0.0):
        raise ValueError("All R values must be positive.")
    sigma = float((config or {}).get("flux", {}).get("sigma", 1.0))
    c = float((config or {}).get("flux", {}).get("c", 0.5))
    ratios = 1.0 + c * (R_vals ** (-sigma))
    return {"R": R_vals, "R_DBI_CM": ratios}
