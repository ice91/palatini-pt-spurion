# palatini_pt/cli.py
# -*- coding: utf-8 -*-
"""
Command-line interface (CLI) stub for Phase 0.2.

Goals
-----
- Provide `palpt --version`
- Provide `palpt figs --which smoke` to verify I/O and headless plotting path.

This uses only stdlib argparse. It tries to generate a tiny smoke figure
if matplotlib is available; otherwise it will still create figs/data and
write a small JSON as a proof-of-run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .io.config import (
    DEFAULT_CONFIG,
    ensure_output_tree,
    load_config,
    write_run_manifest,
    get_pkg_version,
)


def _maybe_smoke_plot(paths) -> Optional[Path]:
    """
    Try to create a tiny smoke plot (PDF + PNG) if matplotlib exists.
    Fallback: return None (no error).
    """
    try:
        # Use non-interactive backend
        import matplotlib

        matplotlib.use("Agg")  # headless safe
        import matplotlib.pyplot as plt

        x = [0, 1, 2, 3, 4]
        y = [0, 1, 0, 1, 0]
        plt.figure(figsize=(3.0, 2.0), dpi=120)
        plt.plot(x, y, linewidth=1.5)
        plt.title("palpt smoke")
        plt.xlabel("x")
        plt.ylabel("y")
        pdf_path = paths.pdf / "fig_smoke.pdf"
        png_path = paths.png / "fig_smoke.png"
        plt.tight_layout()
        plt.savefig(pdf_path)
        plt.savefig(png_path)
        plt.close()
        return pdf_path
    except Exception:
        return None


def _cmd_figs(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else DEFAULT_CONFIG
    out = ensure_output_tree(cfg.get("output", {}).get("dir", "figs"))
    manifest = write_run_manifest(out.data, cfg, extra={"which": args.which})

    # Always write a tiny data stub for the smoke run
    data_stub = out.data / "smoke.json"
    data_stub.write_text(
        json.dumps({"message": "smoke-ok", "which": args.which}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Try to produce a real tiny figure if matplotlib is present
    fig_path = None
    if args.which in ("smoke", "all"):
        fig_path = _maybe_smoke_plot(out)

    print("== palpt figs ==")
    print(f"- which: {args.which}")
    print(f"- output tree: {out.root}")
    print(f"- manifest: {manifest}")
    if fig_path:
        print(f"- smoke figure: {fig_path}")
    else:
        print("- smoke figure: skipped (matplotlib not available)")
    return 0


def _cmd_c1(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else DEFAULT_CONFIG
    out = ensure_output_tree(cfg.get("output", {}).get("dir", "figs"))
    write_run_manifest(out.data, cfg, extra={"phase": "c1", "note": "stub"})
    print("C1 pipeline stub. Implemented in Phase 1.")
    return 0


def _cmd_c2(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else DEFAULT_CONFIG
    out = ensure_output_tree(cfg.get("output", {}).get("dir", "figs"))
    write_run_manifest(out.data, cfg, extra={"phase": "c2", "note": "stub"})
    print("C2 pipeline stub. Implemented in Phase 2.")
    return 0


def _cmd_c3(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else DEFAULT_CONFIG
    out = ensure_output_tree(cfg.get("output", {}).get("dir", "figs"))
    write_run_manifest(out.data, cfg, extra={"phase": "c3", "note": "stub"})
    print("C3 pipeline stub. Implemented in Phase 3.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="palpt",
        description="Palatini × PT-even + spurion — research code CLI (Phase 0.2 stub).",
        add_help=True,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {get_pkg_version()}",
        help="Show package version and exit.",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # figs
    p_figs = sub.add_parser(
        "figs",
        help="Produce figures/data (Phase 0.2 supports 'smoke').",
        description="Generate figures or smoke artifacts. In Phase 0.2, only 'smoke' is supported.",
    )
    p_figs.add_argument(
        "--which",
        type=str,
        default="smoke",
        choices=["smoke", "all"],
        help="What to generate (Phase 0.2: 'smoke' only).",
    )
    p_figs.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON config (optional).",
    )
    p_figs.set_defaults(func=_cmd_figs)

    # c1/c2/c3 stubs (wired now to allow CI smoke later)
    for name, fn, desc in (
        ("c1", _cmd_c1, "Run C1 pipeline (stub for Phase 1)."),
        ("c2", _cmd_c2, "Run C2 pipeline (stub for Phase 2)."),
        ("c3", _cmd_c3, "Run C3 pipeline (stub for Phase 3)."),
    ):
        sp = sub.add_parser(name, help=desc)
        sp.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config (optional).")
        sp.set_defaults(func=fn)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        return int(args.func(args))  # functions already return int
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
