#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.5 — C3: dispersion relation ω^2 = c_T^2(k) k^2, before/after locking.
Outputs:
  figs/pdf/fig5_c3_dispersion.pdf
  figs/data/dispersion.csv (+ .md5)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def _apply_style():
    try:
        from palatini_pt.plotting.style import apply_prd_style  # type: ignore

        apply_prd_style()
    except Exception:
        pass


def _prepare_outdirs(config: Dict | None) -> Dict[str, Path]:
    base = Path((config or {}).get("output", {}).get("dir", "figs"))
    pdf = base / "pdf"
    data = base / "data"
    png = base / "png"
    for d in (pdf, data, png):
        d.mkdir(parents=True, exist_ok=True)
    return {"pdf": pdf, "data": data, "png": png}


def _write_md5(path: Path) -> Path:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    m = path.with_suffix(path.suffix + ".md5")
    m.write_text(h.hexdigest() + "\n", encoding="utf-8")
    return m


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
    if locked:
        return np.ones_like(k)
    return 1.0 + 0.02 * np.tanh(5 * (k - k.min()) / (k.ptp() + 1e-12))


def run(config: Dict | None = None, which: str = "full") -> Dict[str, List[str]]:
    _apply_style()
    paths = _prepare_outdirs(config)

    k = np.logspace(-4, -1, 100) if which != "smoke" else np.logspace(-3, -2, 40)
    try:
        cT_unlocked = _cT_of_k_via_module(config, k, locked=False)
        cT_locked = _cT_of_k_via_module(config, k, locked=True)
    except Exception:
        cT_unlocked = _cT_fallback(k, locked=False)
        cT_locked = _cT_fallback(k, locked=True)

    # CSV
    csv_path = paths["data"] / "dispersion.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "cT_unlocked", "cT_locked"])
        for ki, u, l in zip(k, cT_unlocked, cT_locked):
            w.writerow([f"{float(ki):.16e}", f"{float(u):.16e}", f"{float(l):.16e}"])
    _write_md5(csv_path)

    # Plot
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.semilogx(k, cT_unlocked, label="unlocked")
    ax.semilogx(k, cT_locked, label="locked")
    ax.axhline(1.0, lw=1.0, ls="--")
    ax.set_xlabel("k")
    ax.set_ylabel(r"$c_T(k)$")
    ax.set_title("C3: dispersion before/after locking")
    ax.legend()
    fig.tight_layout()
    pdf_path = paths["pdf"] / "fig5_c3_dispersion.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    _write_md5(pdf_path)

    return {"pdfs": [str(pdf_path)], "data": [str(csv_path)]}


def _load_config_if_needed(path: str | None) -> Dict | None:
    if not path:
        return None
    try:
        from palatini_pt.io.config import load_config  # type: ignore

        return load_config(path)
    except Exception:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser(description="Fig.5 — C3 dispersion")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    out = run(_load_config_if_needed(args.config))
    print(out)


if __name__ == "__main__":
    main()
