# palatini_pt/spurion/profile.py
# -*- coding: utf-8 -*-
"""
Spurion profiles ε(x): value, gradient, and box (d'Alembert/Laplacian).

Phase 1 features
----------------
- ConstantProfile:          ε = const
- PlaneWaveProfile:         ε = A cos(k·x + φ); grad = -A sin(...) k; box = - (k^T g k) A cos(...)
- GaussianRadialProfile:    ε = A exp(-(r/R)^2), r^2 = x^T x
- CustomProfile:            user-defined callables (with finite-diff fallback)

All profiles expose:
    - dim               (int)
    - metric            (np.ndarray)    # g^{μν} contravariant, can be diagonal or full
    - epsilon(x)        -> float
    - grad(x)           -> np.ndarray shape (dim,)
    - box(x)            -> float        # g^{μν} ∂_μ ∂_ν ε

Notes
-----
- Default metric is Euclidean identity of size `dim`.
- For `PlaneWaveProfile`, `box` uses the supplied `metric` exactly: -(k^T g k) * A cos(...).
- For `GaussianRadialProfile`, `box` uses general diagonal or full metric via analytic Hessian.
- For `CustomProfile`, if grad/box are not provided, robust central finite differences are used,
  including cross-derivatives for a full-metric contraction.

This module is dependency-light (NumPy only) and ready for tests in Phase 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

Array = np.ndarray
ArrayLike = Union[Sequence[float], np.ndarray]


# -----------------------------
# Utilities
# -----------------------------


def _as_1d(x: ArrayLike) -> Array:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.ndim != 1:
        raise ValueError("x must be a 1D vector.")
    return x


def _as_metric(g: Optional[ArrayLike], dim: int) -> Array:
    """Accept None | scalar | 1D diag | 2D full and return a (dim, dim) array."""
    if g is None:
        return np.eye(dim, dtype=float)
    G = np.asarray(g, dtype=float)
    if G.ndim == 0:
        return float(G) * np.eye(dim, dtype=float)
    if G.ndim == 1:
        if G.shape[0] != dim:
            raise ValueError(f"diag metric length {G.shape[0]} != dim {dim}")
        return np.diag(G)
    if G.ndim == 2:
        if G.shape != (dim, dim):
            raise ValueError(f"metric shape {G.shape} incompatible with dim {dim}")
        return G
    raise ValueError("metric must be None, scalar, 1D diag, or 2D square.")


def _fd_grad(f: Callable[[Array], float], x: Array, h: float = 1e-6) -> Array:
    """Central finite-diff gradient."""
    n = x.size
    g = np.zeros(n, dtype=float)
    for i in range(n):
        e = np.zeros(n); e[i] = 1.0
        fp = f(x + h * e)
        fm = f(x - h * e)
        g[i] = (fp - fm) / (2.0 * h)
    return g


def _fd_hessian(f: Callable[[Array], float], x: Array, h: float = 1e-5) -> Array:
    """Central finite-diff Hessian (symmetric)."""
    n = x.size
    H = np.zeros((n, n), dtype=float)
    f0 = f(x)
    # Diagonal second derivatives
    for i in range(n):
        e_i = np.zeros(n); e_i[i] = 1.0
        fpp = f(x + h * e_i)
        fmm = f(x - h * e_i)
        H[i, i] = (fpp - 2.0 * f0 + fmm) / (h ** 2)
    # Mixed second derivatives
    for i in range(n):
        e_i = np.zeros(n); e_i[i] = 1.0
        for j in range(i + 1, n):
            e_j = np.zeros(n); e_j[j] = 1.0
            fpp = f(x + h * e_i + h * e_j)
            fpm = f(x + h * e_i - h * e_j)
            fmp = f(x - h * e_i + h * e_j)
            fmm = f(x - h * e_i - h * e_j)
            val = (fpp - fpm - fmp + fmm) / (4.0 * h * h)
            H[i, j] = H[j, i] = val
    return H


# -----------------------------
# Base interface (duck-typed)
# -----------------------------


@dataclass
class BaseProfile:
    dim: int
    metric: Optional[ArrayLike] = None  # g^{μν}

    def __post_init__(self):
        if self.dim <= 0:
            raise ValueError("dim must be positive.")
        self._G = _as_metric(self.metric, self.dim)  # (dim, dim)

    # API to implement
    def epsilon(self, x: ArrayLike) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def grad(self, x: ArrayLike) -> Array:  # pragma: no cover - abstract
        raise NotImplementedError

    def box(self, x: ArrayLike) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    # Convenience
    @property
    def G(self) -> Array:
        """Contravariant metric g^{μν} as a full (dim,dim) array."""
        return self._G


# -----------------------------
# Profiles
# -----------------------------


@dataclass
class ConstantProfile(BaseProfile):
    value: float = 1.0

    def epsilon(self, x: ArrayLike) -> float:
        _ = _as_1d(x)  # validate length if desired
        return float(self.value)

    def grad(self, x: ArrayLike) -> Array:
        x = _as_1d(x)
        return np.zeros(self.dim, dtype=float)

    def box(self, x: ArrayLike) -> float:
        _ = _as_1d(x)
        return 0.0


@dataclass
class PlaneWaveProfile(BaseProfile):
    amplitude: float = 1.0
    kvec: ArrayLike = None  # shape (dim,)
    phase: float = 0.0
    use_cosine: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.kvec is None:
            raise ValueError("PlaneWaveProfile requires kvec.")
        self._k = _as_1d(self.kvec)
        if self._k.size != self.dim:
            raise ValueError(f"kvec length {self._k.size} != dim {self.dim}")

    def _phase(self, x: Array) -> float:
        return float(np.dot(self._k, x) + self.phase)

    def epsilon(self, x: ArrayLike) -> float:
        x = _as_1d(x)
        arg = self._phase(x)
        if self.use_cosine:
            return float(self.amplitude * np.cos(arg))
        else:
            return float(self.amplitude * np.sin(arg))

    def grad(self, x: ArrayLike) -> Array:
        x = _as_1d(x)
        arg = self._phase(x)
        if self.use_cosine:
            # d/dx cos = -sin * k
            return -self.amplitude * np.sin(arg) * self._k
        else:
            # d/dx sin =  cos * k
            return self.amplitude * np.cos(arg) * self._k

    def box(self, x: ArrayLike) -> float:
        x = _as_1d(x)
        arg = self._phase(x)
        # ∂μ∂ν ε = -A cos(arg) kμ kν  (for cosine), or -A sin(arg) kμ kν (for sine)
        if self.use_cosine:
            h_scalar = -self.amplitude * np.cos(arg)
        else:
            h_scalar = -self.amplitude * np.sin(arg)
        # g^{μν} kμ kν
        quad = float(self._k @ self.G @ self._k)
        return float(h_scalar * quad)


@dataclass
class GaussianRadialProfile(BaseProfile):
    amplitude: float = 1.0
    radius: float = 1.0  # R in exp(-(r/R)^2)

    def _r2(self, x: Array) -> float:
        return float(np.dot(x, x))

    def epsilon(self, x: ArrayLike) -> float:
        x = _as_1d(x)
        R2 = self.radius ** 2
        return float(self.amplitude * np.exp(-self._r2(x) / R2))

    def grad(self, x: ArrayLike) -> Array:
        x = _as_1d(x)
        eps = self.epsilon(x)
        return (-2.0 / (self.radius ** 2)) * eps * x

    def hessian(self, x: ArrayLike) -> Array:
        """Analytic Hessian for ε = A exp(-(x·x)/R^2)."""
        x = _as_1d(x)
        n = x.size
        eps = self.epsilon(x)
        R2 = self.radius ** 2
        # ∂i∂j ε = ε * [ (-2/R^2) δij + (4/(R^4)) x_i x_j ]
        H = (-2.0 / R2) * eps * np.eye(n) + (4.0 / (R2 ** 2)) * eps * np.outer(x, x)
        return H

    def box(self, x: ArrayLike) -> float:
        x = _as_1d(x)
        H = self.hessian(x)
        return float(np.einsum("ij,ij->", self.G, H))  # g^{ij} H_{ij}


@dataclass
class CustomProfile(BaseProfile):
    f_epsilon: Callable[[Array], float] = None
    f_grad: Optional[Callable[[Array], Array]] = None
    f_box: Optional[Callable[[Array], float]] = None
    fd_step_grad: float = 1e-6
    fd_step_hess: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if self.f_epsilon is None:
            raise ValueError("CustomProfile requires `f_epsilon` callable.")

    def epsilon(self, x: ArrayLike) -> float:
        x = _as_1d(x)
        return float(self.f_epsilon(x))

    def grad(self, x: ArrayLike) -> Array:
        x = _as_1d(x)
        if self.f_grad is not None:
            g = np.asarray(self.f_grad(x), dtype=float).reshape(-1)
            if g.size != self.dim:
                raise ValueError("Custom grad size mismatch with dim.")
            return g
        # finite-diff
        return _fd_grad(self.f_epsilon, x, h=self.fd_step_grad)

    def box(self, x: ArrayLike) -> float:
        x = _as_1d(x)
        if self.f_box is not None:
            return float(self.f_box(x))
        # Use full Hessian (incl. cross terms) then contract with metric
        H = _fd_hessian(self.f_epsilon, x, h=self.fd_step_hess)
        return float(np.einsum("ij,ij->", self.G, H))


# -----------------------------
# Factory
# -----------------------------


def make(
    kind: str,
    dim: int,
    metric: Optional[ArrayLike] = None,
    **kwargs,
) -> BaseProfile:
    """
    Factory for common spurion profiles.

    Parameters
    ----------
    kind : {"constant","plane","gaussian","custom"}
    dim  : int
    metric : array-like | None
        Contravariant metric g^{μν}. None -> I.

    Returns
    -------
    BaseProfile
    """
    k = kind.lower()
    if k in ("constant", "const"):
        return ConstantProfile(dim=dim, metric=metric, **kwargs)
    if k in ("plane", "planewave", "wave"):
        return PlaneWaveProfile(dim=dim, metric=metric, **kwargs)
    if k in ("gaussian", "radial", "gauss"):
        return GaussianRadialProfile(dim=dim, metric=metric, **kwargs)
    if k in ("custom",):
        return CustomProfile(dim=dim, metric=metric, **kwargs)
    raise ValueError(f"Unknown spurion kind: {kind!r}")


__all__ = [
    "BaseProfile",
    "ConstantProfile",
    "PlaneWaveProfile",
    "GaussianRadialProfile",
    "CustomProfile",
    "make",
]

