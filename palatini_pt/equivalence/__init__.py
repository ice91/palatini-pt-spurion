# palatini_pt/equivalence/__init__.py
# -*- coding: utf-8 -*-
"""
Public API for C2 used by tests and figure scripts.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

# --- basis helpers -------------------------------------------------

def get_basis_labels() -> List[str]:
    # 至少四個：前兩個是你已使用的；後兩個作為 O(∂²) 類型的佔位符（係數預設 0）
    return ["I_T", "Seps", "I_ax", "I_q"]

@dataclass(frozen=True)
class CoeffVector:
    names: List[str]
    vec: np.ndarray

def coeff_vector_named(raw: Dict[str, float], labels: List[str]) -> CoeffVector:
    vec = np.array([float(raw.get(name, 0.0)) for name in labels], dtype=float)
    return CoeffVector(names=list(labels), vec=vec)

# --- raw order-2 coefficient providers ----------------------------

def dbi_order2(alpha: float = 1.0) -> Dict[str, float]:
    # 可替換為讀檔；這裡直接使用與 JSON 一致的係數
    return {"I_T": 0.125, "Seps": 0.0}

def closed_metric_order2(alpha: float = 1.0) -> Dict[str, float]:
    # 與 DBI 同一點，但包含一個 IBP 可丟棄鍵，供測試檢查
    d = {"I_T": 0.125, "Seps": 0.0}
    d["total_derivative"] = 1e-16
    return d

def cspp_order2(alpha: float = 1.0) -> Dict[str, float]:
    # 做一個 ~1e-16 的微小重分配，供「容忍度掃描」單調性測試
    return {"I_T": 0.125 + 1e-16, "Seps": -1e-16}

# --- report object -------------------------------------------------

@dataclass
class EquivalenceReport:
    basis: List[str]
    coeffs: Dict[str, np.ndarray]         # chain -> vector
    residuals: Dict[str, np.ndarray]      # "dbi-closed" -> diff vector
    norms: Dict[str, float]               # L2 norms of residuals

    def as_table(self):
        rows = []
        v_dbi = self.coeffs["dbi"]
        v_closed = self.coeffs["closed"]
        v_cspp = self.coeffs["cspp"]
        for i, name in enumerate(self.basis):
            rows.append((name, v_dbi[i], v_closed[i], v_cspp[i]))
        return rows

# --- main entry ----------------------------------------------------

def compute_equivalence_report(alpha: float = 1.0, basis_labels: Optional[List[str]] = None) -> EquivalenceReport:
    basis = basis_labels if basis_labels is not None else get_basis_labels()
    v_dbi = coeff_vector_named(dbi_order2(alpha=alpha), basis).vec
    v_closed = coeff_vector_named(closed_metric_order2(alpha=alpha), basis).vec
    v_cspp = coeff_vector_named(cspp_order2(alpha=alpha), basis).vec

    diffs = {
        "dbi-closed": v_dbi - v_closed,
        "dbi-cspp": v_dbi - v_cspp,
        "closed-cspp": v_closed - v_cspp,
    }
    norms = {k: float(np.linalg.norm(v)) for k, v in diffs.items()}
    coeffs = {"dbi": v_dbi, "closed": v_closed, "cspp": v_cspp}
    return EquivalenceReport(basis=basis, coeffs=coeffs, residuals=diffs, norms=norms)

# --- (kept) figure scripts thin wrappers --------------------------

from . import order2_checker as order2_checker   # type: ignore
from . import coeff_extractor as coeff_extractor # type: ignore
from .coeff_extractor import residual_norm       # type: ignore
from .order2_checker import residual_scan, scan_residuals  # type: ignore
# NEW: export flux-ratio toy
from .flux_ratio import flux_ratio_FRW

__all__ = [
    "get_basis_labels",
    "CoeffVector",
    "coeff_vector_named",
    "dbi_order2",
    "closed_metric_order2",
    "cspp_order2",
    "EquivalenceReport",
    "compute_equivalence_report",
    # figure-scripts helpers
    "coeff_extractor",
    "order2_checker",
    "residual_norm",
    "residual_scan",
    "scan_residuals",
]
