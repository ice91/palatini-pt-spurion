#!/usr/bin/env python3
"""
Paper artifact: extract (or stage) the 2x2 mixing coefficients used in App. D.

Usage (JSON → CSV + meta + .md5):
  python -m scripts.mixing_matrix_extract \
      --from-json configs/coeffs/mixing_matrix_FRW.json \
      --out-csv   figs/data/mixing_matrix.csv \
      --out-meta  figs/data/mixing_matrix_meta.json \
      --lock-tol  1e-6 \
      --require-lock

JSON schema (minimal):
{
  "N":    { "ROD": <float>, "CM": <float> },
  "divN": { "ROD": <float>, "CM": <float> }
}

This tool:
  1) Writes a normalized CSV: entry,mu_ROD,mu_CM  (entries: N, divN)
  2) Verifies the locking ratio consistency:
       rN   = - mu_CM(N)    / mu_ROD(N)
       rDiv = - mu_CM(divN) / mu_ROD(divN)
     and checks |rN - rDiv| <= tol
  3) Emits a meta JSON with the ratios and w* (locked) up to normalization
  4) Writes .md5 fingerprints for CSV and meta (for reproducibility)
  5) Exits with nonzero code if --require-lock and the ratios mismatch

Notes:
  * If any mu_ROD == 0, locking ratio is ill-defined → error.
  * This script is intentionally lightweight; physics-side extraction
    should produce the JSON. Here we only stage, validate, and log.
"""

from __future__ import annotations
import argparse, json, os, sys, hashlib, time
from typing import Dict, Any, Tuple

try:
    import pandas as pd  # part of dev deps
except Exception:
    sys.stderr.write("Error: pandas not installed. Run 'make install'.\n")
    raise


def _write_md5(path: str) -> None:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    with open(path + ".md5", "w") as g:
        g.write(h.hexdigest() + "\n")


def _need_pair(d: Dict[str, Any], key: str) -> Dict[str, float]:
    if key not in d:
        raise KeyError(f"Missing key '{key}' in JSON.")
    sub = d[key]
    if not all(k in sub for k in ("ROD", "CM")):
        raise KeyError(f"Entry '{key}' must contain 'ROD' and 'CM'.")
    return {"ROD": float(sub["ROD"]), "CM": float(sub["CM"])}


def load_mu_from_json(fp: str) -> Dict[str, Dict[str, float]]:
    with open(fp, "r") as f:
        obj: Dict[str, Any] = json.load(f)
    mu_N    = _need_pair(obj, "N")
    mu_divN = _need_pair(obj, "divN")
    return {"N": mu_N, "divN": mu_divN}


def locking_ratios(mu: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    """Return (rN, rDiv) where r = - mu_CM / mu_ROD for each entry."""
    def r(entry: str) -> float:
        rod = mu[entry]["ROD"]
        cm  = mu[entry]["CM"]
        if abs(rod) == 0.0:
            raise ZeroDivisionError(f"mu_ROD({entry}) == 0 makes locking ratio ill-defined.")
        return - cm / rod
    return r("N"), r("divN")


def write_outputs(mu: Dict[str, Dict[str, float]],
                  out_csv: str,
                  out_meta: str,
                  tol: float,
                  require_lock: bool,
                  print_summary: bool = True) -> int:
    """Write CSV + meta (+md5). Return process exit code (0 ok / 1 fail)."""
    # ensure out dirs
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_meta)) or ".", exist_ok=True)

    # CSV
    df = pd.DataFrame(
        [
            {"entry": "N",    "mu_ROD": mu["N"]["ROD"],    "mu_CM": mu["N"]["CM"]},
            {"entry": "divN", "mu_ROD": mu["divN"]["ROD"], "mu_CM": mu["divN"]["CM"]},
        ]
    )
    df.to_csv(out_csv, index=False)
    _write_md5(out_csv)

    # Ratios & lock check
    rN, rDiv = locking_ratios(mu)
    lock_ok = abs(rN - rDiv) <= tol

    # A canonical locked weight up to normalization (choose w_CM*=1)
    w_star = {
        "w_ROD_star_over_w_CM_star": rN,  # equals rDiv if lock_ok
        "w_ROD_star": rN,
        "w_CM_star": 1.0,
    }

    # Meta JSON
    meta = {
        "timestamp_unix": time.time(),
        "inputs": {
            "mu": mu,
        },
        "outputs": {
            "csv": out_csv,
            "csv_md5": out_csv + ".md5",
            "meta": out_meta,
        },
        "locking_check": {
            "ratio_N": rN,
            "ratio_divN": rDiv,
            "abs_diff": abs(rN - rDiv),
            "tol": tol,
            "lock_ok": lock_ok,
        },
        "locked_weights": w_star,
        "notes": (
            "Locking requires the two ratios to match within tol. "
            "See paper Sec. VI:  w_ROD*/w_CM* = -mu_CM(mu_N)/mu_ROD(mu_N) = -mu_CM(divN)/mu_ROD(divN)."
        ),
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    _write_md5(out_meta)

    if print_summary:
        print("[data] wrote", out_csv, "(+ .md5)")
        print("[meta] wrote", out_meta, "(+ .md5)")
        status = "OK" if lock_ok else "FAIL"
        print(f"[check] rN={rN:.12g}, rDiv={rDiv:.12g}, |Δ|={abs(rN-rDiv):.3g} ≤ {tol} ?  -> {status}")
        if lock_ok:
            print(f"[lock] w*_ROD:w*_CM = {w_star['w_ROD_star']:.12g} : {w_star['w_CM_star']:.12g}")

    if require_lock and not lock_ok:
        return 1
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Write mixing-matrix CSV + meta for App. D (with locking check)")
    ap.add_argument("--from-json",
                    default="configs/coeffs/mixing_matrix_FRW.json",
                    help="JSON file with entries: N/ROD, N/CM, divN/ROD, divN/CM")
    ap.add_argument("--out-csv",
                    default="figs/data/mixing_matrix.csv",
                    help="Output CSV path")
    ap.add_argument("--out-meta",
                    default="figs/data/mixing_matrix_meta.json",
                    help="Output meta JSON path")
    ap.add_argument("--lock-tol", type=float, default=1e-6,
                    help="Tolerance for |ratio(N) - ratio(divN)|")
    ap.add_argument("--require-lock", action="store_true",
                    help="If set, exit with nonzero code when ratios mismatch")
    ap.add_argument("--no-summary", action="store_true",
                    help="If set, suppress summary prints (CI logs can still read meta)")
    args = ap.parse_args()

    mu = load_mu_from_json(args.from_json)
    rc = write_outputs(
        mu=mu,
        out_csv=args.out_csv,
        out_meta=args.out_meta,
        tol=args.lock_tol,
        require_lock=args.require_lock,
        print_summary=not args.no_summary,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
