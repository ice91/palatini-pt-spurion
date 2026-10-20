#!/usr/bin/env python3
"""
Paper artifact: extract (or stage) the 2x2 mixing coefficients used in App. D.

By default, reads a JSON file:
{
  "N":    { "ROD": <float>, "CM": <float> },
  "divN": { "ROD": <float>, "CM": <float> }
}

and writes a normalized CSV to figs/data/mixing_matrix.csv with columns:
entry, mu_ROD, mu_CM
where entry in {"N","divN"}.

Later you can swap --from-json for a compute path that derives mu_* on the fly.
"""

from __future__ import annotations
import argparse, json, os, sys, hashlib
from typing import Dict, Any

try:
    import pandas as pd  # part of dev deps
except Exception as e:  # pragma: no cover
    sys.stderr.write("Error: pandas not installed. Run 'make install'.\n")
    raise

def _write_md5(path: str) -> None:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    with open(path + ".md5", "w") as g:
        g.write(h.hexdigest() + "\n")

def _load_mu_from_json(fp: str) -> Dict[str, Dict[str, float]]:
    with open(fp, "r") as f:
        obj: Dict[str, Any] = json.load(f)

    def _need(d: Dict[str, Any], key: str) -> Dict[str, float]:
        if key not in d:
            raise KeyError(f"Missing key '{key}' in {fp}")
        sub = d[key]
        if not all(k in sub for k in ("ROD", "CM")):
            raise KeyError(f"Entry '{key}' must contain 'ROD' and 'CM'")
        return {"ROD": float(sub["ROD"]), "CM": float(sub["CM"])}

    mu_N    = _need(obj, "N")
    mu_divN = _need(obj, "divN")
    return {"N": mu_N, "divN": mu_divN}

def main() -> None:
    ap = argparse.ArgumentParser(description="Write mixing-matrix CSV for App. D")
    ap.add_argument("--from-json",
                    default="configs/coeffs/mixing_matrix_FRW.json",
                    help="JSON file with entries: N/ROD, N/CM, divN/ROD, divN/CM")
    ap.add_argument("--out",
                    default="figs/data/mixing_matrix.csv",
                    help="Output CSV path")
    args = ap.parse_args()

    mu = _load_mu_from_json(args.from_json)

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(
        [
            {"entry": "N",    "mu_ROD": mu["N"]["ROD"],    "mu_CM": mu["N"]["CM"]},
            {"entry": "divN", "mu_ROD": mu["divN"]["ROD"], "mu_CM": mu["divN"]["CM"]},
        ]
    )
    df.to_csv(args.out, index=False)
    _write_md5(args.out)
    print(f"[data] wrote {args.out} (+ .md5)")

if __name__ == "__main__":
    main()
