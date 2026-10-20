# palatini_pt/algebra/indices.py
# -*- coding: utf-8 -*-
"""
Lightweight tensor/index placeholders for symbolic work at O(∂²).

設計理念
--------
這個模組提供「**指標佔位與常用縮併記號**」的極簡抽象，讓我們能在
`basis.py` / `ibp.py` 以 *index-free* 的組合子（monomials）建構
O(∂²) 的不變式基底，而不先綁死在 heavy 的 `sympy.tensor` 上。

目前約定（可在之後替換成真正的 Tensor 物件）：
- ε : scalar spurion，記號 `eps`
- ∂_μ ε : 以 IndexedBase "d_eps[μ]" 表達；但在基底中用縮寫 monomial 代表
- □ε : 以 Symbol `box_eps` 表達
- T_μ : 輕量扭率（或連接的有效向量模式）以 IndexedBase "T[μ]" 表達
- 縮併 g^{μν} ∂_μ ε ∂_ν ε 直接以 monomial `D` 代表

注意：`basis.py` 只會關心這些**縮寫 monomials**（`D, E, TdotDeps, T2`），
真正的張量細節會在之後（若需要）由 palatini 模組/等效鏈具體化。
"""
from __future__ import annotations

from typing import Dict

import sympy as sp


# -----------------------------
# Scalar / indexed placeholders
# -----------------------------

# Scalar spurion ε（不顯式帶座標）
eps = sp.Symbol("eps", real=True)

# ∂_μ ε 與 T_μ（用 IndexedBase 作占位）
d_eps = sp.IndexedBase("d_eps")  # d_eps[mu]
T_vec = sp.IndexedBase("T")      # T[mu]

# □ε：trace of second derivatives
box_eps = sp.Symbol("box_eps", real=True)  # g^{μν} ∇_μ∇_ν ε（縮寫）


# -----------------------------
# Monomials used by basis.py
# -----------------------------
# O(∂²) 的最小「index-free」基底符號（commutative）
# D      := (∂ε)^2 = g^{μν} ∂_μ ε ∂_ν ε
# E      := ε □ε    （IBP 下 E ≡ -D + total derivative）
# TdotDeps := T^μ ∂_μ ε    （為了 Palatini/扭率與 spurion 的線性耦合）
# T2     := T^2 = g^{μν} T_μ T_ν  （後續用在 algebraic elimination/配分）
D = sp.Symbol("D", real=True)
E = sp.Symbol("E", real=True)
TdotDeps = sp.Symbol("TdotDeps", real=True)
T2 = sp.Symbol("T2", real=True)


def monomial_symbols() -> Dict[str, sp.Symbol]:
    """回傳本模組中所有 index-free monomial 的符號對照。"""
    return {
        "D": D,
        "E": E,
        "TdotDeps": TdotDeps,
        "T2": T2,
    }


__all__ = [
    "eps",
    "d_eps",
    "T_vec",
    "box_eps",
    # monomials
    "D",
    "E",
    "TdotDeps",
    "T2",
    "monomial_symbols",
]
