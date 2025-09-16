#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run all paper figures (Fig.1–7) or a subset (smoke).
Usage:
  python -m scripts.make_all_figs --which all --config configs/default.yaml
  python -m scripts.make_all_figs --which c1|c2|c3|smoke --config ...

This module can also be imported by palpt CLI and call run(which="all"|...).
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Dict, List

# The figure scripts we orchestrate (module order = paper figure order)
_FIG_MODULES = [
    "scripts.fig_c1_pure_trace",
    "scripts.fig_c1_alignment",
    "scripts.fig_c2_coeff_compare",
    "scripts.fig_c3_cT_heatmap",
    "scripts.fig_c3_dispersion",
    "scripts.fig_c3_degeneracy",
    "scripts.fig_gw_waveform_overlay",
]


def _select_modules(which: str) -> List[str]:
    which = which.lower()
    if which in {"all", "full"}:
        return _FIG_MODULES
    if which == "c1":
        return _FIG_MODULES[:2]
    if which == "c2":
        return _FIG_MODULES[2:3]
    if which == "c3":
        return _FIG_MODULES[3:]
    if which == "smoke":
        # 1~2 quick plots for CI
        return [
            "scripts.fig_c2_coeff_compare",
            "scripts.fig_c3_cT_heatmap",
        ]
    # fallback: single module name
    if which.startswith("fig_"):
        return [f"scripts.{which}"]
    raise ValueError(f"Unknown which={which}")


def run(which: str = "all", config: Dict | None = None) -> Dict[str, List[str]]:
    results = {"pdfs": [], "data": []}
    mods = _select_modules(which)
    for modname in mods:
        mod = importlib.import_module(modname)
        if hasattr(mod, "run"):
            out = mod.run(config=config, which="full" if which == "all" else which)
        else:
            raise RuntimeError(f"Module {modname} has no run(config, which) entry point")
        for k in ("pdfs", "data"):
            results[k].extend(out.get(k, []))
    return results


def _load_config_from_file(path: str | Path | None) -> Dict | None:
    if path is None:
        return None
    try:
        from palatini_pt.io.config import load_config  # type: ignore
    except Exception:
        # Very small local YAML loader fallback (no external deps)
        import json
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return raw
    else:
        return load_config(str(path))


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Make all figures")
    p.add_argument("--which", default="all", help="all|c1|c2|c3|smoke|fig_*")
    p.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    args = p.parse_args(argv)

    cfg = _load_config_from_file(args.config)
    out = run(which=args.which, config=cfg)
    print("Generated:")
    for k, v in out.items():
        for x in v:
            print(f"  [{k}] {x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
