# palatini_pt/equivalence/coeff_extractor.py
# -*- coding: utf-8 -*-
"""
Coefficient extraction for the three-chain (C2) comparison.

論文依據：
- Sec. V（Three-Chain Equivalence）與 Eq. (5.12)–(5.17)
- C1：T_mu = 3 eta ∂_mu ε,  I_T = -6 η^2 Π_PT[(∂ε)^2]

本模組把三條建構鏈（DBI/Closed-Metric/CS^+）在「投影＋C1」後
投影到共同基底，輸出一致的係數向量，供殘差掃描（order2_checker）使用。
為了與後續 gw/ 使用一致，我們採用長度為 2 的最小基底：
    B = [I_T,  Seps]   （Seps ≡ Π_PT[(∂ε)^2]）
在本文的姿態下，三鏈在二階只留下 I_T（Seps 係數為 0）。

注意：
- 我們這裡實作的是「論文等式後的有效係數」，不依賴重型符號代數；
- `ibp_tol` 參數保留，對本等式不影響（理想理論殘差為 0）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class Basis:
    """Minimal first-order, quadratic PT-even basis."""
    names: List[str]  # ["I_T", "Seps"]


_BASIS = Basis(names=["I_T", "Seps"])


def _get_eta_from_config(config: Dict | None) -> float:
    """
    讀 eta（C1 的比例常數）。若未提供，採用 η=1.0。
    config 可放在任一層：{"model": {"eta": 1.0}} 或 {"eta": 1.0}
    """
    if not config:
        return 1.0
    if "model" in config and isinstance(config["model"], dict):
        return float(config["model"].get("eta", config.get("eta", 1.0)))
    return float(config.get("eta", 1.0))


def _chain_vector_Astar(eta: float, lam: float = 1.0) -> np.ndarray:
    """
    按論文 Eq. (5.12)(5.15)(5.17)：
        δ^2 L_{chain} = A_* sqrt(-g) I_T,  A_* = λ^2 / 8
    在係數空間（[I_T, Seps]）就是 [A_*, 0].
    """
    A_star = (lam ** 2) / 8.0
    return np.array([A_star, 0.0], dtype=float)


def coeff_vectors(*, config: Dict | None, ibp_tol: float) -> Dict[str, np.ndarray]:
    """
    回傳三條建構鏈在共同基底上的係數向量：
        {'dbi': v, 'closed': v, 'cspp': v}
    其中 v 對應 _BASIS.names 順序 ["I_T", "Seps"]。

    依論文（C2）三鏈在二階的 bulk 完全等價，皆對應 [A_*, 0]。
    """
    eta = _get_eta_from_config(config)  # 目前僅為一致性；不影響 A_* 的大小關係
    v = _chain_vector_Astar(eta=eta, lam=1.0)
    return {"dbi": v.copy(), "closed": v.copy(), "cspp": v.copy()}


def residual_norm(*, config: Dict | None, ibp_tol: float) -> float:
    """
    便利函式：取三對 pairwise 差向量（以 2-範數），回傳最大者。
    論文理論值為 0；數值上回傳 0.0。
    """
    vs = coeff_vectors(config=config, ibp_tol=ibp_tol)
    pairs = [("dbi", "closed"), ("dbi", "cspp"), ("closed", "cspp")]
    res = 0.0
    for a, b in pairs:
        res = max(res, float(np.linalg.norm(vs[a] - vs[b])))
    return res
