# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .coeff_extractor import CoeffVector, coeff_vector_named, get_basis_labels
from .dbi_chain import order2_raw as dbi_order2
from .closed_metric_chain import order2_raw as closed_metric_order2
from .cspp_chain import order2_raw as cspp_order2


@dataclass
class EquivalenceReport:
    """
    C2：三鏈在 O(∂²) 的係數等價性報告。

    Attributes
    ----------
    basis : List[str]
        共同基底順序
    c_dbi : np.ndarray
        DBI 鏈係數向量
    c_closed : np.ndarray
        Closed-metric 鏈係數向量
    c_cspp : np.ndarray
        CS++ 鏈係數向量
    residuals : Dict[str, np.ndarray]
        差向量（例如 'dbi-closed': c_dbi - c_closed）
    norms : Dict[str, float]
        各差向量的 L2 norm
    raw : Dict[str, Dict[str, float]]
        各鏈原始 monomial 係數（尚未丟 IBP 鍵、未對齊）
    """
    basis: List[str]
    c_dbi: np.ndarray
    c_closed: np.ndarray
    c_cspp: np.ndarray
    residuals: Dict[str, np.ndarray]
    norms: Dict[str, float]
    raw: Dict[str, Dict[str, float]]

    def as_table(self) -> List[Tuple[str, float, float, float]]:
        """
        方便印表：[(monomial, c_dbi, c_closed, c_cspp), ...]
        """
        rows: List[Tuple[str, float, float, float]] = []
        for i, name in enumerate(self.basis):
            rows.append((name, float(self.c_dbi[i]), float(self.c_closed[i]), float(self.c_cspp[i])))
        return rows


def _pairwise_residuals(c_dbi: np.ndarray, c_closed: np.ndarray, c_cspp: np.ndarray):
    res = {
        "dbi-closed": c_dbi - c_closed,
        "dbi-cspp": c_dbi - c_cspp,
        "closed-cspp": c_closed - c_cspp,
    }
    norms = {k: float(np.linalg.norm(v)) for k, v in res.items()}
    return res, norms


def compute_equivalence_report(
    alpha: float = 1.0,
    basis_labels: List[str] | None = None,
) -> EquivalenceReport:
    """
    主函式：計算三鏈在 O(∂²) 的係數向量與兩兩殘差。

    Parameters
    ----------
    alpha : float, default 1.0
        鏈內部的示意耦合參數。
    basis_labels : list[str] | None
        指定共同基底順序；若省略，將自動從 `algebra/basis.py` 偵測，或退回預設最小基底。

    Returns
    -------
    EquivalenceReport
    """
    labels = basis_labels if basis_labels is not None else get_basis_labels()

    raw_dbi = dbi_order2(alpha=alpha)
    raw_closed = closed_metric_order2(alpha=alpha)
    raw_cspp = cspp_order2(alpha=alpha)

    v_dbi: CoeffVector = coeff_vector_named(raw_dbi, labels)
    v_closed: CoeffVector = coeff_vector_named(raw_closed, labels)
    v_cspp: CoeffVector = coeff_vector_named(raw_cspp, labels)

    residuals, norms = _pairwise_residuals(v_dbi.vec, v_closed.vec, v_cspp.vec)

    return EquivalenceReport(
        basis=list(labels),
        c_dbi=v_dbi.vec,
        c_closed=v_closed.vec,
        c_cspp=v_cspp.vec,
        residuals=residuals,
        norms=norms,
        raw={"dbi": raw_dbi, "closed": raw_closed, "cspp": raw_cspp},
    )
