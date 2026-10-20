# palatini_pt/numerics/validate.py
# -*- coding: utf-8 -*-
"""
Validation utilities: tolerances, sanity checks, and small diagnostics.

Use cases (Phase 1+)
--------------------
- Near-zero tests with dual thresholds (sym_zero / num_zero)
- Symmetry checks for matrices
- Relative error helpers for reports
- Limit probing (e → 0) using a user callback
- Monotonic convergence checks (e.g., IBP threshold scans later)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np

Array = np.ndarray


# -----------------------------
# Tolerances
# -----------------------------

@dataclass
class Tolerances:
    sym_zero: float = 1.0e-12  # symbolic-level "zero"
    num_zero: float = 1.0e-9   # numerical-level "zero"
    eig_floor: float = 1.0e-12
    rel: float = 1.0e-8


# -----------------------------
# Basic predicates
# -----------------------------

def near_zero(x, *, atol: float) -> bool:
    """True if |x| (or sup-norm of array) <= atol."""
    arr = np.asarray(x, dtype=float)
    return bool(np.max(np.abs(arr)) <= abs(atol))


def zero_mask(x, *, atol: float) -> Array:
    """Elementwise |x| <= atol as bool array."""
    arr = np.asarray(x, dtype=float)
    return np.abs(arr) <= abs(atol)


def assert_near_zero(x, *, atol: float, name: str = "value") -> None:
    if not near_zero(x, atol=atol):
        arr = np.asarray(x, dtype=float)
        raise AssertionError(f"{name} not near zero: max|.|={np.max(np.abs(arr)):.3e} > {atol:.3e}")


def assert_symmetric(A: Array, *, atol: float = 1e-12, name: str = "matrix") -> None:
    A = np.asarray(A, dtype=float)
    dev = np.max(np.abs(A - A.T))
    if dev > atol:
        raise AssertionError(f"{name} not symmetric: max|A-A^T|={dev:.3e} > {atol:.3e}")


def rel_error(a, b, *, mode: str = "max") -> float:
    """Relative error with different denominators.

    mode:
      - "max": |a-b| / max(1, |a|, |b|)
      - "a"  : |a-b| / max(1, |a|)
      - "b"  : |a-b| / max(1, |b|)
      - "mean": |a-b| / max(1, 0.5*(|a|+|b|))
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = np.max(np.abs(a - b))
    if mode == "max":
        den = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    elif mode == "a":
        den = max(1.0, float(np.max(np.abs(a))))
    elif mode == "b":
        den = max(1.0, float(np.max(np.abs(b))))
    elif mode == "mean":
        den = max(1.0, float(0.5 * (np.max(np.abs(a)) + np.max(np.abs(b)))))
    else:
        raise ValueError("Unknown mode for rel_error.")
    return float(num / den)


def assert_allclose(a, b, *, atol: float, rtol: float = 0.0, name: str = "value") -> None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = np.max(np.abs(a - b))
        raise AssertionError(f"{name} not close: max|Δ|={diff:.3e} > atol={atol:.3e} (rtol={rtol:.1e})")


# -----------------------------
# Convergence & limits
# -----------------------------

def is_monotone(xs: Iterable[float], *, kind: str = "nonincreasing", atol: float = 0.0) -> bool:
    """Check monotonic trend up to tolerance."""
    arr = np.asarray(list(xs), dtype=float).reshape(-1)
    if arr.size < 2:
        return True
    dif = np.diff(arr)
    if kind == "nonincreasing":
        return bool(np.all(dif <= atol))
    if kind == "nondecreasing":
        return bool(np.all(dif >= -atol))
    raise ValueError("kind must be 'nonincreasing' or 'nondecreasing'")


def probe_limit_zero(
    f: Callable[[float], float],
    epsilons: Iterable[float],
    *,
    atol: float = 1e-9,
    require_monotone: bool = False,
) -> Tuple[float, bool, bool]:
    """Evaluate f(ε) as ε→0⁺ using a decreasing sequence.

    Returns
    -------
    (last_value, below_atol, monotone_ok)
    """
    eps = np.asarray(list(epsilons), dtype=float).reshape(-1)
    if eps.size == 0:
        raise ValueError("epsilons must be non-empty")
    ys = np.array([float(f(e)) for e in eps], dtype=float)
    last = float(ys[-1])
    below = bool(np.abs(last) <= atol)
    mono_ok = True
    if require_monotone:
        mono_ok = is_monotone(np.abs(ys), kind="nonincreasing", atol=atol * 0.1)
    return last, below, mono_ok


def rms(arr) -> float:
    arr = np.asarray(arr, dtype=float)
    return float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0


__all__ = [
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
