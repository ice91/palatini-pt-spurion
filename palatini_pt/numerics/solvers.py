# palatini_pt/numerics/solvers.py
# -*- coding: utf-8 -*-
"""
Small linear algebra / spectral helpers with NumPy-only core.

Highlights
----------
- Eigendecomposition (symmetric) & spectral utilities
- Generalized symmetric eigenproblem A v = λ B v  (B ≻ 0) via Cholesky
- Stable Cholesky with jitter fallback
- Nullspace (SVD), pseudo-inverse, stable least squares
- PSD checks and projections (for DoF/degeneracy tests later)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

Array = np.ndarray
ArrayLike = np.ndarray | Iterable[float]


# -----------------------------
# Basic helpers
# -----------------------------

def symmetrize(A: Array) -> Array:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


def antisymmetrize(A: Array) -> Array:
    A = np.asarray(A, dtype=float)
    return 0.5 * (A - A.T)


def spectral_radius(A: Array) -> float:
    """ρ(A) = max |λ_i| (not assuming symmetry)."""
    w = np.linalg.eigvals(np.asarray(A, dtype=float))
    return float(np.max(np.abs(w)))


# -----------------------------
# Eigen (symmetric)
# -----------------------------

def eigvals_symmetric(A: Array, *, sort: str = "asc") -> Array:
    """Eigenvalues of a real symmetric matrix."""
    w = np.linalg.eigvalsh(symmetrize(A))
    if sort == "asc":
        return w
    if sort == "desc":
        return w[::-1]
    return w


def eigh_symmetric(A: Array, *, sort: str = "asc") -> Tuple[Array, Array]:
    """(w, V) for symmetric A."""
    w, V = np.linalg.eigh(symmetrize(A))
    if sort == "asc":
        return w, V
    if sort == "desc":
        idx = np.argsort(w)[::-1]
        return w[idx], V[:, idx]
    return w, V


# -----------------------------
# Cholesky (with jitter)
# -----------------------------

@dataclass
class CholeskyResult:
    L: Array
    jitter: float
    attempts: int


def safe_cholesky(A: Array, *, jitter0: float = 1e-12, factor: float = 10.0, max_tries: int = 8) -> CholeskyResult:
    """Try LL^T with increasing jitter*I until success or give up."""
    A = symmetrize(A)
    n = A.shape[0]
    I = np.eye(n, dtype=float)
    jitter = 0.0
    for t in range(max_tries + 1):
        try:
            L = np.linalg.cholesky(A + jitter * I)
            return CholeskyResult(L=L, jitter=jitter, attempts=t + 1)
        except np.linalg.LinAlgError:
            if t == 0:
                jitter = jitter0
            else:
                jitter *= factor
    # last attempt: eigen-based floor
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 0.0)
    Afloor = (V * w) @ V.T
    L = np.linalg.cholesky(Afloor + jitter * I)
    return CholeskyResult(L=L, jitter=jitter, attempts=max_tries + 2)


def is_posdef(A: Array, *, tol: float = 0.0) -> bool:
    """Check positive-definiteness (optionally allow tiny negatives via tol)."""
    try:
        cr = safe_cholesky(A, jitter0=max(tol, 1e-16))
        return True
    except Exception:
        w = eigvals_symmetric(A)
        return bool(np.min(w) > -abs(tol))


def is_semidefinite(A: Array, *, tol: float = 1e-12) -> bool:
    """Check positive-semidefinite within tolerance."""
    w = eigvals_symmetric(A)
    return bool(np.min(w) >= -abs(tol))


def psd_projection(A: Array, *, floor: float = 0.0) -> Array:
    """Project symmetric A to PSD by zeroing negative eigenvalues (≥ floor)."""
    w, V = np.linalg.eigh(symmetrize(A))
    w = np.maximum(w, floor)
    return (V * w) @ V.T


# -----------------------------
# Generalized symmetric eigenproblem
# -----------------------------

def generalized_eigh(A: Array, B: Array) -> Tuple[Array, Array]:
    """Solve A v = λ B v assuming B ≻ 0; returns (w, V) with B-orthonormal V."""
    A = symmetrize(A)
    B = symmetrize(B)
    # Cholesky of B
    chol = safe_cholesky(B)
    L = chol.L
    # transform: A' = L^{-1} A L^{-T}, then eigh
    Linv = np.linalg.inv(L)
    A_tilde = Linv @ A @ Linv.T
    w, Q = np.linalg.eigh(A_tilde)
    # back-transform eigenvectors: V = L^{-T} Q
    V = np.linalg.solve(L.T, Q)
    # Normalize so that V^T B V = I
    # (should already hold numerically; enforce softly)
    for i in range(V.shape[1]):
        norm = np.sqrt(V[:, i].T @ B @ V[:, i])
        if norm > 0:
            V[:, i] /= norm
    return w, V


# -----------------------------
# Linear solves & SVD tools
# -----------------------------

def stable_solve(A: Array, b: Array, *, rcond: float = 1e-12) -> Tuple[Array, float]:
    """Least-squares solver with residual norm; handles rank-deficient."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=rcond)
    res = float(np.sqrt(residuals.sum())) if residuals.size else float(np.linalg.norm(A @ x - b))
    return x, res


def pinv(A: Array, *, rcond: float = 1e-12) -> Array:
    """Moore-Penrose pseudo-inverse."""
    return np.linalg.pinv(np.asarray(A, dtype=float), rcond=rcond)


def nullspace(A: Array, *, rtol: float = 1e-12) -> Array:
    """Right-nullspace basis as columns (possibly empty with shape (n,0))."""
    A = np.asarray(A, dtype=float)
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    tol = rtol * (s[0] if s.size else 1.0)
    mask = s <= tol
    # When full_matrices=True, vh.shape = (n, n). Nullspace vectors correspond to trailing rows.
    # But numerical mask is more robust:
    r = np.sum(~mask)
    ns = vh[r:, :].T  # (n, n-r)
    return ns


def cond_number(A: Array) -> float:
    """2-norm condition number."""
    u, s, vh = np.linalg.svd(np.asarray(A, dtype=float), full_matrices=False)
    if s.size == 0:
        return 0.0
    return float((s[0] / s[-1]) if s[-1] > 0 else np.inf)


def blockdiag(*mats: Array) -> Array:
    """Block diagonal concatenate."""
    mats = [np.asarray(M, dtype=float) for M in mats]
    m = sum(M.shape[0] for M in mats)
    n = sum(M.shape[1] for M in mats)
    out = np.zeros((m, n), dtype=float)
    i = j = 0
    for M in mats:
        h, w = M.shape
        out[i : i + h, j : j + w] = M
        i += h
        j += w
    return out


__all__ = [
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
]
