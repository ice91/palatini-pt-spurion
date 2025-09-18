# palatini_pt/algebra/ibp.py
# -*- coding: utf-8 -*-
"""
IBP (Integration By Parts) rules at O(∂²).

核心想法
--------
在純 spurion 樣式下，O(∂²) 的兩個常見 scalar monomials 為
    D := (∂ε)^2
    E := ε □ε
在丟掉總導數之後有：
    E  ≡  -D    (mod total derivative)

本模組提供：
- `ibp_reduce(expr)`：把含有 E 的表達式規則重寫到 canonical basis（僅含 D）
- `is_total_derivative_free(expr)`：檢查是否已無 E
- `apply_rules(expr, extra)`：允許附加自訂規則
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import sympy as sp

from .indices import D, E, T2, TdotDeps


# -----------------------------
# 基本 IBP 規則
# -----------------------------
# 在 O(∂²) 級別：E -> -D
_DEFAULT_RULES: Tuple[Tuple[sp.Symbol, sp.Expr], ...] = (
    (E, -D),
)


def apply_rules(expr: sp.Expr, extra_rules: Iterable[Tuple[sp.Symbol, sp.Expr]] | None = None) -> sp.Expr:
    """將 expr 依序做符號替換（單向）。"""
    rules: Dict[sp.Symbol, sp.Expr] = dict(_DEFAULT_RULES)
    if extra_rules:
        rules.update(dict(extra_rules))
    out = sp.expand(expr)
    for lhs, rhs in rules.items():
        out = out.xreplace({lhs: rhs})
        out = sp.expand(out)
    return out


def ibp_reduce(expr: sp.Expr) -> sp.Expr:
    """丟總導數後的最小化：E → -D（保留 T·∂ε 與 T² 等不涉及總導數的 monomials）。"""
    return apply_rules(expr, extra_rules=None)


def is_total_derivative_free(expr: sp.Expr) -> bool:
    """檢查是否不再包含 E（即已丟總導數）。"""
    return not expr.has(E)


__all__ = ["apply_rules", "ibp_reduce", "is_total_derivative_free"]

