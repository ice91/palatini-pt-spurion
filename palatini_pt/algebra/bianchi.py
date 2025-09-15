# palatini_pt/algebra/bianchi.py
# -*- coding: utf-8 -*-
"""
Geometric/Bianchi identities (stub layer).

用途
----
C2 與之後的等效鏈化簡可能需要用到幾何恆等式（例如 Riemann/Bianchi 與
扭率相關恆等式）。在本階段（O(∂²) spurion 最小閉環）我們先提供
**介面** 與 **最小模型**，之後若要升級幾何結構，可在此集中擴充。

目前提供：
- `available_identities()`：列出啟用的恆等式名稱（目前僅佔位）
- `apply_bianchi(expr)`：預留掛鉤（現階段對 scalar monomials 無影響）
"""
from __future__ import annotations

from typing import List

import sympy as sp


def available_identities() -> List[str]:
    """列出啟用中的 Bianchi/幾何恆等式（目前僅為佔位，回傳空清單）。"""
    return []


def apply_bianchi(expr: sp.Expr) -> sp.Expr:
    """對表達式套用幾何恆等式（目前為 no-op）。"""
    return expr


__all__ = ["available_identities", "apply_bianchi"]
