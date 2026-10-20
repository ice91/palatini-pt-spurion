from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional

from . import __version__
from .io.config import load_config


def _cmd_figs(args: argparse.Namespace) -> int:
    which = args.which
    if which == "smoke":
        print("[palpt] smoke: no-op demo (Phase 0).")
        return 0
    elif which == "all":
        print("[palpt] 'all' is not implemented yet (will arrive in later phases).")
        return 0
    else:
        print(f"[palpt] unknown figs option: {which}", file=sys.stderr)
        return 2


def _cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_c_placeholder(name: str) -> int:
    print(f"[palpt] '{name}' pipeline will be implemented in later phases.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="palpt",
        description="Palatini × PT-even + spurion CLI (Phase 0 stub)",
    )
    p.add_argument("--config", type=str, default=None, help="YAML/JSON config file")

    sub = p.add_subparsers(dest="subcmd", required=True)

    sp = sub.add_parser("figs", help="Generate figures (smoke or all)")
    sp.add_argument("--which", type=str, choices=["smoke", "all"], default="smoke")
    sp.set_defaults(func=_cmd_figs)

    spv = sub.add_parser("--version", help="Print version", add_help=False)
    spv.set_defaults(func=_cmd_version)

    # pre-create stubs for future phases so CI can call them without failing
    sub.add_parser("c1", help="C1 pipeline (stub)").set_defaults(
        func=lambda _: _cmd_c_placeholder("c1")
    )
    sub.add_parser("c2", help="C2 pipeline (stub)").set_defaults(
        func=lambda _: _cmd_c_placeholder("c2")
    )
    sub.add_parser("c3", help="C3 pipeline (stub)").set_defaults(
        func=lambda _: _cmd_c_placeholder("c3")
    )

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Load config if present (kept for future phases)
    if getattr(args, "config", None):
        try:
            cfg: Dict[str, Any] = load_config(args.config)
            # Silence output in Phase 0; just verify parsing works.
            _ = cfg
        except Exception as e:  # pragma: no cover
            print(f"[palpt] Failed to load config: {e}", file=sys.stderr)
            return 2
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
