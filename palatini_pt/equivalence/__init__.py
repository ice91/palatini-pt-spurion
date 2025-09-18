# palatini_pt/equivalence/__init__.py
# -*- coding: utf-8 -*-
"""
Minimal public API for C2 used by figure scripts.
"""
from . import coeff_extractor, order2_checker
from .coeff_extractor import residual_norm, coeff_vectors, get_basis_labels
from .order2_checker import residual_scan, scan_residuals

__all__ = [
    "coeff_extractor",
    "order2_checker",
    "residual_norm",
    "coeff_vectors",
    "get_basis_labels",
    "residual_scan",
    "scan_residuals",
]
