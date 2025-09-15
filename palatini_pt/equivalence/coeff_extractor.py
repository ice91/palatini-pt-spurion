# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# 嘗試讀取使用者在 algebra/basis.py 定義的共同基底順序；
# 若不存在就使用此預設最小基底。
DEFAULT_BASIS: List[str] = [
    "d_eps_sq",  # (∂ε)^2
    "box_eps",  # □ε
    "torsion_trace_dot_grad_eps",  # T·∂ε
    "torsion_trace_sq",  # T^2
]

# 會在 IBP 階段被直接忽略的鍵（總導數、Bianchi 恒等式後為 0 等）
IBP_IGNORED_KEYS: Tuple[str, ...] = ("total_derivative",)


def _try_import_user_basis() -> List[str] | None:
    """
    嘗試從 palatini_pt.algebra.basis 載入使用者定義的基底順序。

    允許下列命名其中之一：
      - get_basis_labels() -> Sequence[str]
      - basis_labels()     -> Sequence[str]
      - BASIS_LABELS       : Sequence[str]
      - MINIMAL_BASIS      : Sequence[str]
    """
    try:
        from palatini_pt.algebra import basis as user_basis  # type: ignore
    except Exception:
        return None

    for name in ("get_basis_labels", "basis_labels"):
        fn = getattr(user_basis, name, None)
        if fn is not None:
            try:
                labels = list(fn())  # type: ignore[call-arg]
                if labels:
                    return labels
            except Exception:
                pass

    for name in ("BASIS_LABELS", "MINIMAL_BASIS"):
        labels = getattr(user_basis, name, None)
        if isinstance(labels, (list, tuple)) and len(labels) > 0:
            return list(labels)
    return None


def get_basis_labels() -> List[str]:
    """
    取得共同基底順序。若使用者未提供，回傳 DEFAULT_BASIS。
    """
    labels = _try_import_user_basis()
    return labels if labels is not None else list(DEFAULT_BASIS)


@dataclass(frozen=True)
class CoeffVector:
    """
    共同基底上的係數向量包裝。

    Attributes
    ----------
    labels : List[str]
        基底順序（monomial 名稱）。
    vec : np.ndarray
        對齊後的係數向量，shape=(n_basis, )
    raw : Dict[str, float]
        原始 monomial 係數字典（未對齊、含可能被忽略的鍵）。
    """

    labels: List[str]
    vec: np.ndarray
    raw: Dict[str, float]


def _filter_ibp_keys(raw: Dict[str, float]) -> Dict[str, float]:
    """移除 IBP 可忽略的鍵（例如 'total_derivative'）。"""
    return {k: v for k, v in raw.items() if k not in IBP_IGNORED_KEYS}


def coeff_vector_named(
    raw: Dict[str, float],
    basis_labels: Sequence[str] | None = None,
) -> CoeffVector:
    """
    將「monomial->係數」字典對齊到指定基底的係數向量。

    缺少的 monomial 自動補 0；多餘的 monomial 被保留在 `raw`，但不會進向量。
    """
    labels = list(basis_labels) if basis_labels is not None else get_basis_labels()
    filtered = _filter_ibp_keys(raw)
    vec = np.zeros(len(labels), dtype=float)
    pos = {name: i for i, name in enumerate(labels)}
    for k, v in filtered.items():
        if k in pos:
            vec[pos[k]] = float(v)
        else:
            # 不在基底中的 monomial（例如更高階）保持忽略狀態
            pass
    return CoeffVector(labels=labels, vec=vec, raw=raw)
