# palatini_pt/spurion/pt_even.py
# -*- coding: utf-8 -*-
"""
PT-even selector for monomials built from basic building blocks.

Design goals (Phase 1)
----------------------
- Minimal, explicit parity bookkeeping (±1) for named atoms.
- Parse simple multiplicative expressions with optional exponents:
    "grad_eps^2 * eps", "box * eps", "d * d * eps", "g * grad_eps * grad_eps"
- Provide:
    - PTRegistry: registry of atom → parity (±1), extensible.
    - compute_parity(expr): +1 or -1 (strict by default).
    - pt_project(exprs): keep only PT-even.
    - assert_pt_even(exprs): assert all are PT-even, else raise with details.

Conventions (defaults; can be changed via registry):
----------------------------------------------------
- "eps" / "epsilon"        : +1   (spurion scalar)
- "d", "partial", "nabla"  : -1   (derivative operator under PT)
- "grad_eps"               : -1   (one derivative acting on eps)
- "box", "lap_eps"         : +1   (two derivatives on eps contracted)
- "g"                      : +1   (metric tensor)
You can register extra atoms (e.g., "torsion_T") as needed.

These rules are self-consistent for typical O(∂^2) invariants:
- (grad_eps)^2  → (-1)^2 = +1  (PT-even)
- box * eps     → (+1)*(+1) = +1
- d * eps       → -1            (PT-odd, will be projected out)

This module is algebra-agnostic: it does not depend on SymPy or the rest
of the pipeline; it just classifies & filters strings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


# -----------------------------
# Registry
# -----------------------------


@dataclass
class PTRegistry:
    """
    Mapping from atom name to PT parity (+1 or -1).

    Parameters
    ----------
    mapping : dict[str, int] | None
        Seed mapping; if None, defaults are used.

    Methods
    -------
    register(name: str, parity: int)
    parity(name: str) -> int
    """
    mapping: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.mapping:
            self.mapping.update(
                {
                    "eps": +1,
                    "epsilon": +1,
                    "d": -1,
                    "partial": -1,
                    "nabla": -1,
                    "grad_eps": -1,
                    "lap_eps": +1,
                    "box": +1,
                    "g": +1,
                }
            )

    def register(self, name: str, parity: int) -> None:
        if parity not in (+1, -1):
            raise ValueError("parity must be ±1.")
        self.mapping[name] = int(parity)

    def parity(self, name: str) -> int:
        if name not in self.mapping:
            raise KeyError(f"Unknown atom '{name}'. Register it or pass strict=False.")
        return int(self.mapping[name])


# -----------------------------
# Expression parsing
# -----------------------------

_TOKEN = r"[A-Za-z_][A-Za-z0-9_\-]*"
_TERM_RE = re.compile(rf"\s*({_TOKEN})(?:\s*\^\s*([+-]?\d+))?\s*")


def _parse_expr(expr: str) -> List[Tuple[str, int]]:
    """
    Parse a simple product expression into [(name, power), ...].

    Grammar (multiplicative only):
        EXPR := TERM ( ('*'|whitespace) TERM )*
        TERM := NAME [ '^' INT ]

    Examples
    --------
    "grad_eps^2 * eps"    -> [("grad_eps", 2), ("eps", 1)]
    "d * d * eps"         -> [("d", 1), ("d", 1), ("eps", 1)]
    "box*epsilon"         -> [("box", 1), ("epsilon", 1)]
    """
    s = expr.strip()
    if not s:
        raise ValueError("Empty expression.")
    # split by '*' OR whitespace; we re-scan term by term
    parts = [p for p in re.split(r"\*", s) if p.strip()]
    tokens: List[Tuple[str, int]] = []
    for part in parts:
        # a part may still contain whitespace-separated terms; tokenize greedily
        pos = 0
        text = part.strip()
        while pos < len(text):
            m = _TERM_RE.match(text, pos)
            if not m:
                # try to consume a single whitespace and continue
                if text[pos].isspace():
                    pos += 1
                    continue
                raise ValueError(f"Cannot parse around: {text[pos:pos+16]!r}")
            name = m.group(1)
            pow_str = m.group(2)
            power = int(pow_str) if pow_str is not None else 1
            tokens.append((name, power))
            pos = m.end()
    return tokens


# -----------------------------
# Parity evaluation
# -----------------------------


def compute_parity(
    expr: str | Sequence[Tuple[str, int]],
    registry: PTRegistry | None = None,
    *,
    strict: bool = True,
    unknown_default: int = +1,
) -> int:
    """
    Compute PT parity of an expression.

    Parameters
    ----------
    expr : str | list[(name, power)]
        Multiplicative expression (see _parse_expr).
    registry : PTRegistry | None
        Parity registry (uses defaults if None).
    strict : bool
        If True, unknown atom raises. If False, use `unknown_default`.
    unknown_default : int
        Used when strict=False and atom not in registry.

    Returns
    -------
    +1 or -1
    """
    reg = registry or PTRegistry()
    terms = _parse_expr(expr) if isinstance(expr, str) else list(expr)
    parity = +1
    for name, power in terms:
        if strict:
            p = reg.parity(name)
        else:
            p = reg.mapping.get(name, unknown_default)
            if p not in (+1, -1):
                p = unknown_default
        # parity multiplies; negative power flips accordingly but we reduce mod 2
        if power % 2 != 0:
            parity *= p
    return +1 if parity >= 0 else -1


def is_pt_even(expr: str | Sequence[Tuple[str, int]], registry: PTRegistry | None = None) -> bool:
    return compute_parity(expr, registry=registry) == +1


def pt_project(
    exprs: Iterable[str | Sequence[Tuple[str, int]]],
    registry: PTRegistry | None = None,
) -> List[str]:
    """
    Keep only PT-even expressions. Returns them as strings (round-tripped if needed).
    """
    reg = registry or PTRegistry()
    kept: List[str] = []
    for e in exprs:
        s = e if isinstance(e, str) else " * ".join(f"{n}^{p}" if p != 1 else n for n, p in e)
        if is_pt_even(e, registry=reg):
            kept.append(s)
    return kept


def assert_pt_even(
    exprs: Iterable[str | Sequence[Tuple[str, int]]],
    registry: PTRegistry | None = None,
) -> None:
    """
    Assert all expressions are PT-even. Raises AssertionError listing offenders.
    """
    reg = registry or PTRegistry()
    odd: List[str] = []
    for e in exprs:
        s = e if isinstance(e, str) else " * ".join(f"{n}^{p}" if p != 1 else n for n, p in e)
        if not is_pt_even(e, registry=reg):
            odd.append(s)
    if odd:
        raise AssertionError("Found PT-odd monomials (project them out): " + ", ".join(odd))


__all__ = [
    "PTRegistry",
    "compute_parity",
    "is_pt_even",
    "pt_project",
    "assert_pt_even",
]

