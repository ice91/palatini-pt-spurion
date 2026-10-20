# -*- coding: utf-8 -*-
"""
palatini_pt.equivalence
=======================

C2：三條建構鏈（DBI / closed-metric / CS++）在 O(∂²) 的係數等價性工具。
提供：
- 各鏈 O(∂²) 原始係數（字典）
- 對齊共同基底的係數向量
- 等價性殘差報告 (Order-2 Equivalence)
"""
from .dbi_chain import order2_raw as dbi_order2
from .closed_metric_chain import order2_raw as closed_metric_order2
from .cspp_chain import order2_raw as cspp_order2

from .coeff_extractor import (
    get_basis_labels,
    coeff_vector_named,
    CoeffVector,
)

from .order2_checker import (
    EquivalenceReport,
    compute_equivalence_report,
)

__all__ = [
    "dbi_order2",
    "closed_metric_order2",
    "cspp_order2",
    "get_basis_labels",
    "coeff_vector_named",
    "CoeffVector",
    "EquivalenceReport",
    "compute_equivalence_report",
]
