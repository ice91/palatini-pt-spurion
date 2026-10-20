# palatini_pt/algebra/basis.py
# -*- coding: utf-8 -*-
"""
O(∂²) invariant basis construction and projection.

最小可用版本 — 特色
-------------------
- 以 **index-free monomials** 表示 O(∂²) 基底：
    D := (∂ε)^2
    E := ε □ε       (IBP 下 E ≡ -D)
    TdotDeps := T·∂ε
    T2 := T^2
- 提供「**canonical basis**」= [D, TdotDeps, T2]
  （E 會在 IBP 後消去）
- `project_to_canonical(expr)`：把輸入的 SymPy 線性組合（允許含 E）
  投影到 canonical 係數向量（NumPy 1-D）
- `projection_matrix(from_basis, to_basis)`：回傳線性投影矩陣
- `basis_info()`：回報基底與規則摘要

之後若要擴充：
- 可以在本模組加上 O(∂²) 其他允許項（含 connection/contorsion 的縮寫 monomials），
  或升級到 `sympy.tensor` 並以 index-aware 規則自動化化簡。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import sympy as sp

from .indices import D, E, T2, TdotDeps, monomial_symbols
from .ibp import ibp_reduce
from .bianchi import apply_bianchi


Symbol = sp.Symbol
Expr = sp.Expr
Array = np.ndarray


# -----------------------------
# 定義「canonical」與「擴充」基底
# -----------------------------
# canonical basis（IBP 後不含 E）
CANONICAL: Tuple[Symbol, ...] = (D, TdotDeps, T2)

# 一個包含 E 的「擴充」基底（外部鏈條／原始展開常用）
EXTENDED: Tuple[Symbol, ...] = (D, E, TdotDeps, T2)


@dataclass(frozen=True)
class Basis:
    """不變式基底包裝。

    Parameters
    ----------
    monomials : Sequence[Symbol]
        基底 monomials 的順序定義。
    name : str
        基底名稱（僅供描述）。
    """
    monomials: Tuple[Symbol, ...]
    name: str = "custom"

    def as_list(self) -> List[Symbol]:
        return list(self.monomials)

    def index(self, s: Symbol) -> int:
        return self.monomials.index(s)

    def size(self) -> int:
        return len(self.monomials)


CANONICAL_BASIS = Basis(CANONICAL, name="canonical")
EXTENDED_BASIS = Basis(EXTENDED, name="extended")


# -----------------------------
# 工具：取係數向量 / 建式
# -----------------------------

def _coeff_vector(expr: sp.Expr, basis: Basis) -> np.ndarray:
    expr = sp.expand(expr)
    vec = np.zeros((basis.size(),), dtype=object)  # object！
    for i, sym in enumerate(basis.monomials):
        coeff = sp.expand(expr).coeff(sym)
        vec[i] = coeff  # 不轉 float
        expr = sp.expand(expr - coeff * sym)
    return vec

def _from_coeff_vector(coeffs: np.ndarray, basis: Basis) -> sp.Expr:
    if coeffs.shape != (basis.size(),):
        raise ValueError(...)
    out = 0
    for ci, sym in zip(coeffs, basis.monomials):
        if ci:
            out += ci * sym  # 不轉 float
    return sp.expand(out)


# -----------------------------
# 投影矩陣（線性代數觀點）
# -----------------------------

def projection_matrix(from_basis: Basis, to_basis: Basis) -> np.ndarray:
    P = np.zeros((to_basis.size(), from_basis.size()), dtype=object)  # object！
    for j, sym in enumerate(from_basis.monomials):
        expr_j = ibp_reduce(apply_bianchi(sym))
        vec_to = _coeff_vector(expr_j, to_basis)
        P[:, j] = vec_to
    return P


# -----------------------------
# 高階 API：把任意 expr → canonical 係數
# -----------------------------

def project_to_canonical(expr: Expr) -> Array:
    """先套用（幾何恆等式 + IBP），再讀取 canonical 的係數向量。"""
    expr_red = ibp_reduce(apply_bianchi(sp.expand(expr)))
    return _coeff_vector(expr_red, CANONICAL_BASIS)


def basis_info() -> Dict[str, object]:
    """回傳基底與規則摘要（方便 CLI/除錯列印）。"""
    sym_map = {k: str(v) for k, v in monomial_symbols().items()}
    return {
        "canonical_order": [str(s) for s in CANONICAL_BASIS.monomials],
        "extended_order": [str(s) for s in EXTENDED_BASIS.monomials],
        "ibp_rules": {"E ->": "-D"},
        "symbols": sym_map,
        "notes": [
            "E ≡ -D (mod total derivative) at O(∂²).",
            "TdotDeps, T2 保留為 spurion–torsion 的有效耦合縮寫，後續由 palatini 決定係數。",
        ],
    }


__all__ = [
    "Basis",
    "CANONICAL_BASIS",
    "EXTENDED_BASIS",
    "projection_matrix",
    "project_to_canonical",
    "basis_info",
]
