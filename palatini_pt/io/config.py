# palatini_pt/io/config.py
# -*- coding: utf-8 -*-
"""
Minimal config I/O utilities for the Palatini × PT-even + spurion project.

Features
--------
- load_config(): Load YAML or JSON, or fall back to a sane DEFAULT_CONFIG.
- ensure_output_tree(): Create figs/{pdf,png,data} tree if absent.
- write_run_manifest(): Persist run metadata (git hash, versions, timestamp).
- safe imports and graceful degradation (no hard deps beyond stdlib).

This module is intentionally minimal for Phase 0.2; later phases can extend it
(e.g. add schema validation, md5 snapshotting, richer manifest).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

# -----------------------------
# Defaults (kept tiny for 0.2)
# -----------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "dtype": "float64",
    "tolerances": {"sym_zero": 1.0e-12, "num_zero": 1.0e-9},
    "grids": {
        "ct": {
            "k": [1e-3],  # minimal smoke grid
            "param1": {"min": -1.0, "max": 1.0, "n": 11},
            "param2": {"min": -1.0, "max": 1.0, "n": 11},
        }
    },
    "spurion": {"kind": "constant", "omega": 0.0},
    "output": {"dir": "figs", "keep_intermediate": True},
}


# -----------------------------
# Helpers
# -----------------------------


def _probe_yaml_loader():
    """Return (yaml_module | None)."""
    try:
        import yaml  # type: ignore

        return yaml
    except Exception:
        return None


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def get_git_hash(cwd: Optional[Path] = None) -> Optional[str]:
    """Return short git commit hash if repo present; else None."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(cwd or Path.cwd())
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def get_pkg_version(dist_name: str = "palatini-pt-spurion") -> str:
    """
    Best-effort package version discovery.

    Priority:
      1) importlib.metadata.version("palatini-pt-spurion")
      2) __version__ attribute on palatini_pt package
      3) "0.0.0+local"
    """
    # Try canonical dist name
    try:
        return metadata.version(dist_name)
    except Exception:
        pass
    # Try module attribute
    try:
        from palatini_pt import __version__  # type: ignore

        return str(__version__)
    except Exception:
        return "0.0.0+local"


# -----------------------------
# Public API
# -----------------------------


def load_config(path: Optional[str | os.PathLike] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a YAML/JSON configuration file. If `path` is None or file not found,
    return a deepcopy-like of DEFAULT_CONFIG (we copy shallow dicts).

    Parameters
    ----------
    path : str | Path | None
        Path to YAML (.yml/.yaml) or JSON (.json) config file.
    overrides : dict | None
        Optional dictionary to shallow-merge into the loaded/base config.

    Returns
    -------
    dict
        Merged configuration.
    """
    cfg: Dict[str, Any] = json.loads(_json_dumps(DEFAULT_CONFIG))  # cheap deep copy

    if path is not None:
        p = Path(path)
        if p.exists() and p.is_file():
            suffix = p.suffix.lower()
            try:
                if suffix in (".yml", ".yaml"):
                    yaml = _probe_yaml_loader()
                    if yaml is None:
                        raise RuntimeError(
                            f"Cannot load YAML config '{p}': PyYAML is not installed. "
                            "Install with `pip install pyyaml` or use JSON."
                        )
                    cfg_file = yaml.safe_load(_read_text(p)) or {}
                elif suffix == ".json":
                    cfg_file = json.loads(_read_text(p))
                else:
                    # Try YAML first, then JSON
                    yaml = _probe_yaml_loader()
                    if yaml is not None:
                        try:
                            cfg_file = yaml.safe_load(_read_text(p)) or {}
                        except Exception:
                            cfg_file = json.loads(_read_text(p))
                    else:
                        cfg_file = json.loads(_read_text(p))
                if not isinstance(cfg_file, dict):
                    raise ValueError(f"Config file must be a mapping, got {type(cfg_file)}")
                # shallow merge
                cfg.update(cfg_file)
            except Exception as e:
                raise RuntimeError(f"Failed to load config from '{p}': {e}") from e

    if overrides:
        cfg.update(overrides)
    return cfg


@dataclass
class OutputTree:
    root: Path
    pdf: Path
    png: Path
    data: Path


def ensure_output_tree(base_dir: str | os.PathLike) -> OutputTree:
    """
    Ensure figs/{pdf,png,data} exists under `base_dir` and return paths.

    Parameters
    ----------
    base_dir : str | Path
        Base output directory (e.g., 'figs').

    Returns
    -------
    OutputTree
        Paths to root/pdf/png/data.
    """
    root = Path(base_dir)
    pdf = root / "pdf"
    png = root / "png"
    data = root / "data"
    for d in (root, pdf, png, data):
        d.mkdir(parents=True, exist_ok=True)
    return OutputTree(root=root, pdf=pdf, png=png, data=data)


def write_run_manifest(
    output_dir: str | os.PathLike,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
    dist_name: str = "palatini-pt-spurion",
) -> Path:
    """
    Write run manifest JSON with git hash, versions, timestamp, and config.

    Parameters
    ----------
    output_dir : str | Path
        Directory where 'run_manifest.json' will be written (commonly figs/data).
    config : dict
        The effective config used by the run.
    extra : dict | None
        Any additional info to include (e.g. CLI args).
    dist_name : str
        Distribution name for version lookup.

    Returns
    -------
    Path
        Path to the written JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "project": "palatini-pt-spurion",
        "version": get_pkg_version(dist_name),
        "python": sys.version.split()[0],
        "git": {"hash": get_git_hash()},
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": config,
    }
    if extra:
        payload["extra"] = extra

    out_path = output_dir / "run_manifest.json"
    out_path.write_text(_json_dumps(payload), encoding="utf-8")
    return out_path
