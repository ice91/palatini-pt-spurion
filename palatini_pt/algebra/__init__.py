# palatini_pt/algebra/__init__.py
# -*- coding: utf-8 -*-
"""
Algebra layer: indices/placeholders, IBP rules, Bianchi (stub), and O(∂²) basis.
"""
from .indices import (
    eps,
    d_eps,
    T_vec,
    box_eps,
    D,
    E,
    TdotDeps,
    T2,
    monomial_symbols,
)
from .ibp import apply_rules, ibp_reduce, is_total_derivative_free
from .bianchi import available_identities, apply_bianchi
from .basis import (
    Basis,
    CANONICAL_BASIS,
    EXTENDED_BASIS,
    projection_matrix,
    project_to_canonical,
    basis_info,
)

__all__ = [
    # indices
    "eps",
    "d_eps",
    "T_vec",
    "box_eps",
    "D",
    "E",
    "TdotDeps",
    "T2",
    "monomial_symbols",
    # ibp
    "apply_rules",
    "ibp_reduce",
    "is_total_derivative_free",
    # bianchi
    "available_identities",
    "apply_bianchi",
    # basis
    "Basis",
    "CANONICAL_BASIS",
    "EXTENDED_BASIS",
    "projection_matrix",
    "project_to_canonical",
    "basis_info",
]

