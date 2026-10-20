# palatini_pt/equivalence/__init__.py
# -*- coding: utf-8 -*-
"""
palatini_pt.equivalence — minimal public API for C2 used by figure scripts.
This init avoids importing non-existent names so that submodules can be imported.
"""

from . import coeff_extractor as coeff_extractor
from . import order2_checker as order2_checker

# （可選）順手 re-export 常用函式，方便外部直接 from ... import 使用
from .coeff_extractor import residual_norm as residual_norm
from .coeff_extractor import compare_residual_norm as compare_residual_norm
from .order2_checker import residual_scan as residual_scan
from .order2_checker import scan_residuals as scan_residuals

__all__ = [
    "coeff_extractor",
    "order2_checker",
    "residual_norm",
    "compare_residual_norm",
    "residual_scan",
    "scan_residuals",
]
