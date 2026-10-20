# palatini_pt/equivalence/cspp_chain.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

def order2_raw(config: Dict | None = None) -> Dict[str, float]:
    path = (config or {}).get("coeffs", {}).get("cspp", "configs/coeffs/cspp.json")
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"I_T": float(d["I_T"]), "Seps": float(d.get("Seps", 0.0))}
