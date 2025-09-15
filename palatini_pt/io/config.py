from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load YAML/JSON config file.

    Parameters
    ----------
    path : str | Path
        Path to a YAML (.yml/.yaml) or JSON (.json) file.

    Returns
    -------
    dict
        Parsed configuration as Python dict.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If file extension is not supported, or content is invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix in {".yml", ".yaml"}:
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported config type: {suffix}")

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/dict.")

    return data
