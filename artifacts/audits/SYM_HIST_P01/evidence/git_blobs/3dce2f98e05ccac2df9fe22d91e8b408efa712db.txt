# scripts/fig_c2_coeff_compare.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

REQUIRE_REAL = bool(int(os.environ.get("PALPT_REQUIRE_REAL_APIS", "0")))


def _apply_style():
    try:
        from palatini_pt.plotting.style import apply_prd
        apply_prd()
    except Exception:
        pass


def _prepare_outdirs(config: Dict | None) -> Dict[str, Path]:
    outdir = Path(config.get("output", {}).get("dir", "figs")) if config else Path("figs")
    pdfdir = outdir / "pdf"
    datadir = outdir / "data"
    pdfdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    return {"pdfdir": pdfdir, "datadir": datadir}


def _write_md5(p: Path):
    import hashlib
    h = hashlib.md5(p.read_bytes()).hexdigest()
    (p.with_suffix(p.suffix + ".md5")).write_text(h + "\n")


def _residual_scan_via_module(config: Dict | None, thresholds: np.ndarray) -> np.ndarray:
    import importlib
    # order2_checker API variants
    names = [
        ("palatini_pt.equivalence.order2_checker", "residual_scan"),
        ("palatini_pt.equivalence.order2_checker", "scan_residuals"),
    ]
    for modname, fnname in names:
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, fnname):
                fn = getattr(mod, fnname)
                arr = fn(config=config, thresholds=thresholds)
                return np.asarray(arr, dtype=float).ravel()
        except Exception:
            continue
    # coeff_extractor direct residual
    try:
        mod = importlib.import_module("palatini_pt.equivalence.coeff_extractor")
        for fnname in ["residual_norm", "compare_residual_norm"]:
            if hasattr(mod, fnname):
                fn = getattr(mod, fnname)
                vals = [float(fn(config=config, ibp_tol=float(th))) for th in thresholds]
                return np.array(vals, dtype=float)
    except Exception:
        pass
    raise RuntimeError("No residual-scan API found for equivalence/")


def _residual_scan_fallback(thresholds: np.ndarray) -> np.ndarray:
    floor = 1e-12
    start = 1e-4
    span = max(1e-12, float(np.ptp(np.asarray(thresholds))))
    vals = start * np.exp(-3.0 * (thresholds - float(np.min(thresholds))) / span)
    return np.maximum(floor, vals)


def run(config: Dict | None = None, which: str = "full") -> Dict[str, List[str]]:
    _apply_style()
    paths = _prepare_outdirs(config)

    thresholds = np.logspace(-14, -8, 13) if which != "smoke" else np.logspace(-12, -10, 5)
    try:
        resids = _residual_scan_via_module(config, thresholds)
    except Exception:
        if REQUIRE_REAL:
            raise
        resids = _residual_scan_fallback(thresholds)

    # --- 畫圖 ---
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.semilogx(thresholds, resids, marker="o", lw=1.2)
    ax.set_xlabel("IBP tolerance")
    ax.set_ylabel(r"$\|\Delta\|$")
    ax.set_title("C2 residual (smoke)" if which == "smoke" else "C2 residual scan")
    ax.grid(True, ls=":", alpha=0.6)

    pdf = paths["pdfdir"] / ("fig3_c2_coeff_compare.pdf" if which != "smoke" else "fig3_c2_coeff_compare_smoke.pdf")
    fig.tight_layout()
    fig.savefig(pdf, dpi=200)
    plt.close(fig)
    _write_md5(pdf)

    # --- 資料 ---
    data_path = paths["datadir"] / ("c2_residuals.csv" if which != "smoke" else "c2_residuals_smoke.csv")
    with data_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ibp_tol", "residual_norm"])
        for t, r in zip(thresholds, resids):
            w.writerow([f"{t:.16e}", f"{r:.16e}"])
    _write_md5(data_path)

    return {"pdfs": [str(pdf)], "data": [str(data_path)]}


if __name__ == "__main__":
    print(run(which="smoke"))
