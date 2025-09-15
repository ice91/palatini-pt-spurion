# palatini_pt/numerics/grids.py
# -*- coding: utf-8 -*-
"""
Numerical grids and sweeps.

Features
--------
- Reproducible RNG seeding: set_seed / get_rng
- Param grid from config-like dicts (values list or {min,max,n} lin/log)
- Named Cartesian products → stacked 2D arrays or list of dicts
- Chunked iteration for large sweeps

Typical config (from YAML via io.config):
    grids:
      ct:
        k: [1e-4, 1e-3, 1e-2, 1e-1]                # explicit values
        param1: {min: -1.0, max: 1.0, n: 61}      # linear
        param2: {min: -2, max: 0, n: 5, scale: log10}  # log10 space
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

Array = np.ndarray
ArrayLike = Union[Sequence[float], np.ndarray]


# -----------------------------
# RNG (reproducible)
# -----------------------------

_RNG: Optional[np.random.Generator] = None


def set_seed(seed: int | None) -> None:
    """Set a global numpy Generator for reproducibility (or None → default)."""
    global _RNG
    _RNG = np.random.default_rng(None if seed is None else int(seed))


def get_rng() -> np.random.Generator:
    """Get the global Generator (lazily created if unset)."""
    global _RNG
    if _RNG is None:
        _RNG = np.random.default_rng()
    return _RNG


# -----------------------------
# Grid specs
# -----------------------------

@dataclass
class GridSpec:
    """A single-parameter grid specification.

    Either provide explicit `values`, or provide (`min_`, `max_`, `n`) and optional `scale`.

    Parameters
    ----------
    values : Sequence[float] | np.ndarray | None
        Explicit points.
    min_ , max_ : float | None
        Range bounds (used if values is None).
    n : int | None
        Number of points (used if values is None).
    scale : {"lin","log10"}
        Spacing type (linspace or 10**linspace in exponent).
    dtype : np.dtype | str
        Output dtype, default float64.
    """
    values: Optional[ArrayLike] = None
    min_: Optional[float] = None
    max_: Optional[float] = None
    n: Optional[int] = None
    scale: str = "lin"  # or "log10"
    dtype: np.dtype | str = np.float64

    def materialize(self) -> Array:
        if self.values is not None:
            arr = np.asarray(self.values, dtype=self.dtype).reshape(-1)
            if arr.ndim != 1:
                raise ValueError("GridSpec.values must be 1-D")
            return arr
        if self.min_ is None or self.max_ is None or self.n is None:
            raise ValueError("GridSpec requires either `values` or (min_, max_, n)")
        if self.n <= 0:
            raise ValueError("GridSpec.n must be positive")
        if self.scale.lower() in ("lin", "linear", "linspace"):
            arr = np.linspace(self.min_, self.max_, int(self.n), dtype=self.dtype)
        elif self.scale.lower() in ("log10", "log", "logspace"):
            # interpret bounds as exponents (base 10)
            exps = np.linspace(self.min_, self.max_, int(self.n), dtype=self.dtype)
            arr = np.power(10.0, exps, dtype=self.dtype)
        else:
            raise ValueError(f"Unknown scale {self.scale!r}")
        return arr


def _parse_param_spec(obj: object, dtype=np.float64) -> GridSpec:
    """Parse a parameter spec from a value/list/dict."""
    if isinstance(obj, dict):
        # keys could be min/max/n/scale
        min_ = obj.get("min", obj.get("min_", None))
        max_ = obj.get("max", obj.get("max_", None))
        n = obj.get("n", None)
        scale = obj.get("scale", "lin")
        return GridSpec(values=None, min_=min_, max_=max_, n=n, scale=scale, dtype=dtype)
    # list/array/scalar → explicit values
    vals = np.asarray(obj, dtype=dtype).reshape(-1)
    return GridSpec(values=vals, dtype=dtype)


def params_from_dict(d: Mapping[str, object], *, dtype=np.float64) -> Dict[str, Array]:
    """Turn a {name: spec} dict into {name: 1D array}."""
    out: Dict[str, Array] = {}
    for key, spec in d.items():
        out[key] = _parse_param_spec(spec, dtype=dtype).materialize()
    return out


# -----------------------------
# Cartesian products
# -----------------------------

def named_product(grid_dict: Mapping[str, Array]) -> List[Dict[str, float]]:
    """List of dicts for each grid point."""
    keys = list(grid_dict.keys())
    vals = [np.asarray(grid_dict[k]).reshape(-1) for k in keys]
    combos = []
    for tup in product(*vals):
        combos.append({k: float(v) for k, v in zip(keys, tup)})
    return combos


def stack_product(grid_dict: Mapping[str, Array], *, order: Optional[Sequence[str]] = None) -> Tuple[Array, List[str]]:
    """Stacked 2D array of shape (N, P) with parameter order."""
    if order is None:
        order = list(grid_dict.keys())
    cols = [np.asarray(grid_dict[k]).reshape(-1) for k in order]
    N = int(np.prod([c.size for c in cols], dtype=int))
    P = len(cols)
    out = np.empty((N, P), dtype=float)
    # fill by iterating product
    idx = 0
    for tup in product(*cols):
        out[idx, :] = np.asarray(tup, dtype=float)
        idx += 1
    return out, list(order)


def meshgrid_dict(grid_dict: Mapping[str, Array]) -> Dict[str, Array]:
    """Return broadcastable meshgrids in a dict, like np.meshgrid but named."""
    keys = list(grid_dict.keys())
    arrays = [np.asarray(grid_dict[k]).reshape(-1) for k in keys]
    meshes = np.meshgrid(*arrays, indexing="ij")
    return {k: m for k, m in zip(keys, meshes)}


# -----------------------------
# Iteration helpers
# -----------------------------

def iter_chunks(indices_or_array: ArrayLike, chunk_size: int) -> Iterator[Array]:
    """Yield consecutive chunks of indices/rows."""
    arr = np.asarray(indices_or_array)
    n = arr.shape[0]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for start in range(0, n, chunk_size):
        yield arr[start : start + chunk_size]


def batched_named_product(grid_dict: Mapping[str, Array], *, batch: int) -> Iterator[List[Dict[str, float]]]:
    """Named product but yielded in batches for large sweeps."""
    lst = named_product(grid_dict)
    for i in range(0, len(lst), batch):
        yield lst[i : i + batch]


__all__ = [
    "GridSpec",
    "set_seed",
    "get_rng",
    "params_from_dict",
    "named_product",
    "stack_product",
    "meshgrid_dict",
    "iter_chunks",
    "batched_named_product",
]
