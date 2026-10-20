# palatini_pt/equivalence/coeff_extractor.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from .dbi_chain import order2_raw as dbi_order2
from .closed_metric_chain import order2_raw as closed_order2
from .cspp_chain import order2_raw as cspp_order2

@dataclass(frozen=True)
class Basis:
    names: List[str]  # ["I_T", "Seps"]

_BASIS = Basis(names=["I_T", "Seps"])

def get_basis_labels() -> List[str]:
    return list(_BASIS.names)

def _vec_from_dict(d: Dict[str, float]) -> np.ndarray:
    return np.array([float(d.get("I_T", 0.0)), float(d.get("Seps", 0.0))], dtype=float)

def coeff_vectors(*, config: Dict | None, ibp_tol: float) -> Dict[str, np.ndarray]:
    # 這裡假設 JSON 已是「投影＋IBP」後的係數
    v_dbi = _vec_from_dict(dbi_order2(config))
    v_clo = _vec_from_dict(closed_order2(config))
    v_csp = _vec_from_dict(cspp_order2(config))
    return {"dbi": v_dbi, "closed": v_clo, "cspp": v_csp}

def residual_norm(*, config: Dict | None, ibp_tol: float) -> float:
    vs = coeff_vectors(config=config, ibp_tol=ibp_tol)
    pairs = [("dbi", "closed"), ("dbi", "cspp"), ("closed", "cspp")]
    return float(max(np.linalg.norm(vs[a] - vs[b]) for a, b in pairs))

# --- symbolic helper for notebooks ---
from typing import Any

try:
    import sympy as sp  # optional; used when notebooks pass sympy symbols
except Exception:  # pragma: no cover
    sp = None

def reduce_to_IT(*, route: str, lam: Any, eta: Any, Seps: Any):
    """
    Return the quadratic reduction A_* * I_T as a symbolic/arith expression.

    Parameters
    ----------
    route : {'dbi', 'closed_metric', 'cspp', 'closed'}
        Which chain to use. 'closed' is accepted as alias of 'closed_metric'.
    lam, eta, Seps : numbers or sympy symbols
        Book-keeping λ, alignment η (>0), and projected Seps ≡ Π_PT[(∂ε)^2].

    Returns
    -------
    expr : same numeric type as inputs (sympy if provided)
        A_* * I_T with A_* = lam**2 / 8 and I_T = -6 * eta**2 * Seps.
    """
    key = route.lower().replace("-", "_")
    if key not in {"dbi", "closed_metric", "cspp", "closed"}:
        raise ValueError(f"unknown route: {route!r}")
    # alias
    if key == "closed":
        key = "closed_metric"

    # A_* is identical for DBI / closed-metric / CS^{+} at quadratic order
    A_star = (lam ** 2) / 8

    # I_T under C1: pure-trace torsion ⇒ I_T = -6 η^2 Seps
    I_T = -6 * (eta ** 2) * Seps

    if sp is not None and any(isinstance(x, sp.Basic) for x in (lam, eta, Seps)):
        return sp.simplify(A_star * I_T)
    return A_star * I_T
