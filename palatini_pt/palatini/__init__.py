# palatini_pt/palatini/__init__.py
# -*- coding: utf-8 -*-
"""
Palatini layer: torsion/connection utilities and C1 pipeline.
"""
from .connection import minkowski_metric, torsion_from_trace, contorsion_from_trace
from .torsion_decomp import (
    inverse_metric,
    trace_vector,
    axial_vector,
    pure_trace_piece,
    axial_piece,
    decompose,
)
from .field_eq import C1Solution, solve_torsion_from_spurion, check_pure_trace
from .c1_pure_trace import alignment_angle, run_c1, smoke_example, C1Report

"""Palatini sector (C1)."""
from . import c1_pure_trace as c1_pure_trace  # re-export
from . import field_eq as field_eq            # re-export
#__all__ = ["c1_pure_trace", "field_eq"]

__all__ = [
    # connection
    "minkowski_metric",
    "torsion_from_trace",
    "contorsion_from_trace",
    # decomp
    "inverse_metric",
    "trace_vector",
    "axial_vector",
    "pure_trace_piece",
    "axial_piece",
    "decompose",
    # field eq
    "C1Solution",
    "solve_torsion_from_spurion",
    "check_pure_trace",
    # C1 driver
    "alignment_angle",
    "run_c1",
    "smoke_example",
    "C1Report",
    "c1_pure_trace",
    "field_eq",
]

