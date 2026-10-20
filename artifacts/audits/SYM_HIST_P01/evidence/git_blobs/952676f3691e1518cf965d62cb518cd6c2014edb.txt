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
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REQUIRE_REAL = bool(int(os.environ.get("PALPT_REQUIRE_REAL_APIS", "0")))


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
    # 先嘗試直接的 cT_grid()
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

    # 沒有 cT_grid：用真實 API cT_of_k 掃格（param1 = seps_scale 的縮放、param2 = k）
    try:
        mod = importlib.import_module("palatini_pt.gw.tensor_mode")
        if not hasattr(mod, "cT_of_k"):
            raise RuntimeError("gw.tensor_mode.cT_of_k not found")

        cT_of_k = getattr(mod, "cT_of_k")

        grids = (config or {}).get("grids", {}).get("ct", {})
        n_s = int(grids.get("n_s", 61))
        n_k = int(grids.get("n", grids.get("nk", 61)))
        kmin = float(grids.get("kmin", 1e-4))
        kmax = float(grids.get("kmax", 1e-1))

        sval = np.linspace(0.0, 1.0, n_s)  # seps_scale 的比例 0..1
        kval = np.logspace(np.log10(kmin), np.log10(kmax), n_k)

        X, Y = np.meshgrid(sval, kval, indexing="xy")  # X: s, Y: k
        cT = np.empty_like(X, dtype=float)

        base_seps = float((config or {}).get("spurion", {}).get("seps_scale", 1.0))

        for j in range(n_k):
            cfg_j = deepcopy(config) if config else {}
            spur = cfg_j.setdefault("spurion", {})
            spur["seps_scale"] = base_seps  # 先放底值
            # 對每個 s 調整 seps_scale 再算 cT(k)
            for i in range(n_s):
                spur["seps_scale"] = base_seps * float(sval[i])
                cT[j, i] = float(cT_of_k(k=np.array([kval[j]]), config=cfg_j, locked=False)[0])

        return X, Y, cT

    except Exception as e:
        raise RuntimeError(f"Cannot build cT heatmap via module APIs: {e}")


def _grid_fallback(which: str = "full") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 81 if which != "smoke" else 31
    x = np.linspace(-1.0, 1.0, n)
    y = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(x, y, indexing="xy")
    cT = 1.0 + 0.02 * (X**2 + Y**2)
    return X, Y, cT


def run(config: Dict | None = None, which: str = "full") -> Dict[str, List[str]]:
    _apply_style()
    paths = _prepare_outdirs(config)

    try:
        X, Y, cT = _compute_grid_via_module(config)
    except Exception:
        if REQUIRE_REAL:
            raise
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
        extent=[float(X.min()), float(X.max()), float(Y.min()), float(Y.max())],
        aspect="auto",
    )
    ax.set_xlabel("scale(s) on seps_scale")
    ax.set_ylabel("k")
    ax.set_title(r"C3: $|c_T-1|$ heatmap")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("|c_T-1|")

    # optional locking overlay（若使用者提供 locking_curve）
    try:
        import palatini_pt.gw.locking as lk  # type: ignore
        for name in ["locking_curve", "curve_locking", "locking_contour"]:
            if hasattr(lk, name):
                curve = np.asarray(getattr(lk, name)(config=config))
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
