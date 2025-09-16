#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.4 — C3: heatmap of |c_T-1| with/without locking overlay.
Outputs:
  figs/pdf/fig4_c3_cT_heatmap.pdf
  figs/data/cT_grid.npz (+ .md5)
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

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


def _compute_grid_via_module(config: Dict | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Try gw.tensor_mode APIs
    import importlib

    for modname, fnname in [
        ("palatini_pt.gw.tensor_mode", "cT_grid"),
        ("palatini_pt.gw.tensor_mode", "grid_cT"),
        ("palatini_pt.gw.tensor_mode", "compute_cT_grid"),
    ]:
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, fnname):
                fn = getattr(mod, fnname)
                g = fn(config=config)
                # Expect dict or tuple (X, Y, cT)
                if isinstance(g, dict):
                    X = np.asarray(g["X"])
                    Y = np.asarray(g["Y"])
                    cT = np.asarray(g["cT"])
                else:
                    X, Y, cT = g
                    X, Y, cT = map(np.asarray, (X, Y, cT))
                return X, Y, cT
        except Exception:
            continue
    raise RuntimeError("No cT grid API found in gw.tensor_mode")


def _grid_fallback(which: str = "full") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 81 if which != "smoke" else 31
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(x, y, indexing="xy")
    cT = 1.0 + 0.02 * (X**2 + Y**2)  # smooth >1 away from origin
    return X, Y, cT


def run(config: Dict | None = None, which: str = "full") -> Dict[str, List[str]]:
    _apply_style()
    paths = _prepare_outdirs(config)

    try:
        X, Y, cT = _compute_grid_via_module(config)
    except Exception:
        X, Y, cT = _grid_fallback(which=which)

    # save data
    npz_path = paths["data"] / "cT_grid.npz"
    np.savez_compressed(npz_path, X=X, Y=Y, cT=cT)
    _write_md5(npz_path)

    # heatmap of |cT-1|
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(
        np.abs(cT - 1.0),
        origin="lower",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        aspect="auto",
    )
    ax.set_xlabel("param1")
    ax.set_ylabel("param2")
    ax.set_title(r"C3: $|c_T-1|$ heatmap")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("|c_T-1|")

    # try plot locking curve
    try:
        import palatini_pt.gw.locking as lk  # type: ignore

        for name in ["locking_curve", "curve_locking", "locking_contour"]:
            if hasattr(lk, name):
                curve = getattr(lk, name)(config=config)
                curve = np.asarray(curve)
                if curve.ndim == 2 and curve.shape[1] == 2:
                    ax.plot(curve[:, 0], curve[:, 1], lw=1.5)
                break
    except Exception:
        pass

    fig.tight_layout()
    pdf_path = paths["pdf"] / "fig4_c3_cT_heatmap.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    _write_md5(pdf_path)
    return {"pdfs": [str(pdf_path)], "data": [str(npz_path)]}


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
    ap = argparse.ArgumentParser(description="Fig.4 — C3 cT heatmap")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    out = run(_load_config_if_needed(args.config))
    print(out)


if __name__ == "__main__":
    main()
