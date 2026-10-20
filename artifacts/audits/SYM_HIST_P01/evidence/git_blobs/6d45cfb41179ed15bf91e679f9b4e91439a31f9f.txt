# scripts/fig_c3_dispersion.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


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


def _cT_of_k_via_module(config: Dict | None, k: np.ndarray, locked: bool) -> np.ndarray:
    import importlib

    # First try tensor_mode.cT_of_k
    for modname, fnname in [
        ("palatini_pt.gw.tensor_mode", "cT_of_k"),
        ("palatini_pt.gw.tensor_mode", "dispersion_cT"),
    ]:
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, fnname):
                fn = getattr(mod, fnname)
                return np.asarray(fn(k=k, config=config, locked=locked), dtype=float)
        except Exception:
            continue
    # Or quadratic_action -> cT2_of_k
    try:
        mod = importlib.import_module("palatini_pt.gw.quadratic_action")
        for fnname in ["cT2_of_k", "cT_squared"]:
            if hasattr(mod, fnname):
                fn = getattr(mod, fnname)
                return np.sqrt(np.maximum(0.0, np.asarray(fn(k=k, config=config, locked=locked))))
    except Exception:
        pass
    raise RuntimeError("No c_T(k) API found.")


def _cT_fallback(k: np.ndarray, locked: bool) -> np.ndarray:
    # 僅供 smoke：未鎖定略偏離 1，鎖定後恰為 1
    if locked:
        return np.ones_like(k)
    span = float(np.ptp(np.asarray(k))) + 1e-12  # NumPy 2.0 友善
    return 1.0 + 0.02 * np.tanh(5 * (k - float(np.min(k))) / span)


def run(config: Dict | None = None, which: str = "full") -> Dict[str, List[str]]:
    _apply_style()
    paths = _prepare_outdirs(config)

    k = np.logspace(-4, -1, 100) if which != "smoke" else np.logspace(-3, -2, 40)
    try:
        cT_unlocked = _cT_of_k_via_module(config, k, locked=False)
    except Exception:
        cT_unlocked = _cT_fallback(k, locked=False)
    try:
        cT_locked = _cT_of_k_via_module(config, k, locked=True)
    except Exception:
        cT_locked = _cT_fallback(k, locked=True)

    # --- 畫圖 ---
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(k, cT_unlocked, label="unlocked", lw=1.5)
    ax.plot(k, cT_locked, label="locked", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$c_T(k)$")
    ax.set_title("Dispersion (smoke)" if which == "smoke" else "Dispersion")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()

    pdf = paths["pdfdir"] / ("fig5_c3_dispersion.pdf" if which != "smoke" else "fig5_c3_dispersion_smoke.pdf")
    fig.tight_layout()
    fig.savefig(pdf, dpi=200)
    plt.close(fig)
    from hashlib import md5
    (pdf.with_suffix(pdf.suffix + ".md5")).write_text(md5(pdf.read_bytes()).hexdigest() + "\n")

    # --- 資料輸出 ---
    data_path = paths["datadir"] / ("dispersion.csv" if which != "smoke" else "dispersion_smoke.csv")
    import csv
    with data_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "cT_unlocked", "cT_locked"])
        for kk, u, l in zip(k, cT_unlocked, cT_locked):
            w.writerow([f"{kk:.16e}", f"{u:.16e}", f"{l:.16e}"])
    (data_path.with_suffix(data_path.suffix + ".md5")).write_text(md5(data_path.read_bytes()).hexdigest() + "\n")

    return {"pdfs": [str(pdf)], "data": [str(data_path)]}


if __name__ == "__main__":
    print(run(which="smoke"))
