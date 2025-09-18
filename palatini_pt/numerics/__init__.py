# palatini_pt/numerics/__init__.py
# -*- coding: utf-8 -*-
"""
Numerics toolbox (grids / solvers / validation).
"""
from .grids import (
    GridSpec,
    set_seed,
    get_rng,
    params_from_dict,
    named_product,
    stack_product,
    meshgrid_dict,
    iter_chunks,
    batched_named_product,
)
from .solvers import (
    symmetrize,
    antisymmetrize,
    spectral_radius,
    eigvals_symmetric,
    eigh_symmetric,
    safe_cholesky,
    CholeskyResult,
    is_posdef,
    is_semidefinite,
    psd_projection,
    generalized_eigh,
    stable_solve,
    pinv,
    nullspace,
    cond_number,
    blockdiag,
)
from .validate import (
    Tolerances,
    near_zero,
    zero_mask,
    assert_near_zero,
    assert_symmetric,
    rel_error,
    assert_allclose,
    is_monotone,
    probe_limit_zero,
    rms,
)

__all__ = [
    # grids
    "GridSpec",
    "set_seed",
    "get_rng",
    "params_from_dict",
    "named_product",
    "stack_product",
    "meshgrid_dict",
    "iter_chunks",
    "batched_named_product",
    # solvers
    "symmetrize",
    "antisymmetrize",
    "spectral_radius",
    "eigvals_symmetric",
    "eigh_symmetric",
    "safe_cholesky",
    "CholeskyResult",
    "is_posdef",
    "is_semidefinite",
    "psd_projection",
    "generalized_eigh",
    "stable_solve",
    "pinv",
    "nullspace",
    "cond_number",
    "blockdiag",
    # validate
    "Tolerances",
    "near_zero",
    "zero_mask",
    "assert_near_zero",
    "assert_symmetric",
    "rel_error",
    "assert_allclose",
    "is_monotone",
    "probe_limit_zero",
    "rms",
]

