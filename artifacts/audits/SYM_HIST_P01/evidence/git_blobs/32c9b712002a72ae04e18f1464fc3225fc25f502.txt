#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig.7 — GW waveform overlay: GR vs model (locked).
Outputs:
  figs/pdf/fig7_gw_waveform_overlay.pdf
  figs/data/waveform_overlay.csv (+ .md5)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
from typing import Dict, List

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


def _waveforms_via_module(config: Dict | None, t: np.ndarray) -> Dict[str, np.ndarray]:
    import importlib
    for modname, fnname in [
        ("palatini_pt.gw.tensor_mode", "waveform_overlay"),
        ("palatini_pt.gw.tensor_mode", "waveforms_gr_model"),
    ]:
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, fnname):
                fn = getattr(mod, fnname)
                d = fn(config=config) if "t" not in fn.__code__.co_varnames else fn(t=t, config=config)
                # 相容不同鍵名
                tt = np.asarray(d.get("t", t))
                gr = d.get("h_GR", d.get("gr"))
                md = d.get("h_model", d.get("model"))
                if gr is None or md is None:
                    raise RuntimeError("waveform dict keys not found")
                return {"t": tt, "gr": np.asarray(gr), "model": np.asarray(md)}
        except Exception:
            continue
    # 用 cT_of_k 建構簡單波包（locked）
    try:
        mod = importlib.import_module("palatini_pt.gw.tensor_mode")
        if hasattr(mod, "cT_of_k"):
            k = 0.05
            cT = float(getattr(mod, "cT_of_k")(k=np.array([k]), config=config, locked=True)[0])
            gr = np.sin(k * t)
            model = np.sin(cT * k * t)
            return {"t": t, "gr": gr, "model": model}
    except Exception:
        pass
    raise RuntimeError("No waveform API found.")


def _waveforms_fallback(t: np.ndarray) -> Dict[str, np.ndarray]:
    gr = np.sin(0.2 * t)
    model = np.sin(0.2 * t + 0.01 * np.sin(0.05 * t))
    return {"t": t, "gr": gr, "model": model}


def run(config: Dict | None = None, which: str = "full") -> Dict[str, List[str]]:
    _apply_style()
    paths = _prepare_outdirs(config)

    t = np.linspace(0.0, 200.0, 4000) if which != "smoke" else np.linspace(0.0, 80.0, 800)
    try:
        d = _waveforms_via_module(config, t)
    except Exception:
        if REQUIRE_REAL:
            raise
        d = _waveforms_fallback(t)

    # CSV
    csv_path = paths["data"] / "waveform_overlay.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "h_GR", "h_model"])
        for ti, gi, mi in zip(d["t"], d["gr"], d["model"]):
            w.writerow([f"{float(ti):.16e}", f"{float(gi):.16e}", f"{float(mi):.16e}"])
    _write_md5(csv_path)

    # Plot
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(d["t"], d["gr"], label="GR")
    ax.plot(d["t"], d["model"], label="Model (locked)")
    ax.set_xlabel("t")
    ax.set_ylabel("h")
    ax.set_title("GW waveform overlay")
    ax.legend()
    fig.tight_layout()
    pdf_path = paths["pdf"] / "fig7_gw_waveform_overlay.pdf"
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
    ap = argparse.ArgumentParser(description="Fig.7 — GW waveform overlay")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    out = run(_load_config_if_needed(args.config))
    print(out)


if __name__ == "__main__":
    main()
