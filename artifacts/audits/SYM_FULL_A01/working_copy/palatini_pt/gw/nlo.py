# palatini_pt/gw/nlo.py
# -*- coding: utf-8 -*-
"""
NLO (dim-6) PT-even corrections to the quadratic tensor sector.

We parametrize two representative, projector-preserving, first-derivative-per-factor,
PT-even effective structures at next-to-leading order:
  - gradT_sq_eff           ~ (∇ T)^2 - type effective contribution after C1 map
  - Ricci_deps_deps_eff    ~ R_{μν} ∂^μ ε ∂^ν ε - type effective structure

At the locked point (K=G=1 at LO), these NLO pieces induce
  ΔK(k) ~ a * k^2 / Λ^2
  ΔG(k) ~ (a + b) * k^2 / Λ^2
so that δc_T^2(k) = ΔG - ΔK = b * k^2 / Λ^2.

This file provides a small, testable parametrization and simple helpers for figures.
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


# Minimal PT-even dim-6 “basis” labels (ASCII to keep tooling happy)
_BASIS = ["gradT_sq_eff", "Ricci_deps_deps_eff"]


def basis_labels_nlo() -> list[str]:
    """Return the list of NLO PT-even basis labels used in this minimal model."""
    return list(_BASIS)


def coeffs_from_config(config: Dict | None) -> Dict[str, float]:
    """
    Read NLO coefficients from config under key 'nlo'.
    Expected:
        {"nlo": {
            "gradT_sq_eff": <float>,           # default 0.0
            "Ricci_deps_deps_eff": <float>,    # default 0.0
            "Lambda2": <float>                  # Λ^2 scale; default 1e6
        }}
    """
    c = (config or {}).get("nlo", {})
    out = {
        "gradT_sq_eff": float(c.get("gradT_sq_eff", 0.0)),
        "Ricci_deps_deps_eff": float(c.get("Ricci_deps_deps_eff", 0.0)),
    }
    return out


def _lambda2_from_config(config: Dict | None) -> float:
    """Helper to read Λ^2; default to a large value."""
    return float((config or {}).get("nlo", {}).get("Lambda2", 1e6))


def delta_KG(k: np.ndarray, config: Dict | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (ΔK(k), ΔG(k)) at NLO in a simple k^2/Λ^2 model.

    Model:
        a ≡ gradT_sq_eff / Λ^2
        b ≡ Ricci_deps_deps_eff / Λ^2
        ΔK(k) = a * k^2
        ΔG(k) = (a + b) * k^2

    Parameters
    ----------
    k : array_like
        Wavenumbers (same units as used in your dispersion figures).
    config : dict or None
        See coeffs_from_config() docstring.

    Returns
    -------
    (dK, dG) : Tuple[np.ndarray, np.ndarray]
    """
    k = np.asarray(k, dtype=float)
    cs = coeffs_from_config(config)
    lam2 = _lambda2_from_config(config)
    a = cs["gradT_sq_eff"] / lam2
    b = cs["Ricci_deps_deps_eff"] / lam2
    dK = a * (k ** 2)
    dG = (a + b) * (k ** 2)
    return dK, dG


def predict_offsets(k: np.ndarray, config: Dict | None = None) -> Dict[str, np.ndarray]:
    """
    Convenience wrapper returning a dict of offsets:
        {'delta_cT2': ΔG-ΔK, 'delta_K': ΔK, 'delta_G': ΔG}
    """
    dK, dG = delta_KG(k, config)
    return {"delta_cT2": dG - dK, "delta_K": dK, "delta_G": dG}
