# tests/test_scripts_figs.py
# -*- coding: utf-8 -*-
"""
Smoke tests for figure scripts in scripts/*.py

這些測試：
- 以 which="smoke" 呼叫每張圖的 run(config, which) 入口；
- 將輸出導向 pytest 的 tmp_path/figs；
- 驗證至少一個 PDF 與一個 data 檔存在，且對應 .md5 存在；
- 另測 scripts.make_all_figs.run(which="smoke") 的整體流程。
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Dict, List

import pytest


def _cfg(tmp_path: Path) -> Dict:
    return {
        "output": {
            "dir": str(tmp_path / "figs")
        }
    }


@pytest.mark.parametrize(
    "modname",
    [
        "scripts.fig_c1_pure_trace",
        "scripts.fig_c1_alignment",
        "scripts.fig_c2_coeff_compare",
        "scripts.fig_c3_cT_heatmap",
        "scripts.fig_c3_dispersion",
        "scripts.fig_c3_degeneracy",
        "scripts.fig_gw_waveform_overlay",
    ],
)
def test_each_fig_module_smoke(modname: str, tmp_path: Path):
    mod = importlib.import_module(modname)
    out = mod.run(config=_cfg(tmp_path), which="smoke")
    # 至少有一個 pdf 與一個 data
    assert isinstance(out, dict)
    pdfs: List[str] = out.get("pdfs", [])
    data: List[str] = out.get("data", [])
    assert len(pdfs) >= 1, f"{modname} should output at least 1 pdf"
    assert len(data) >= 1, f"{modname} should output at least 1 data file"
    # 檔案存在且有 .md5
    for p in pdfs + data:
        pth = Path(p)
        assert pth.exists(), f"missing output: {pth}"
        md5 = pth.with_suffix(pth.suffix + ".md5")
        assert md5.exists(), f"missing md5 for: {pth}"


def test_make_all_figs_smoke(tmp_path: Path):
    mod = importlib.import_module("scripts.make_all_figs")
    out = mod.run(which="smoke", config=_cfg(tmp_path))
    # 應至少產生我們定義的 smoke 圖（Fig.3 + Fig.4）
    pdfs = [Path(p) for p in out.get("pdfs", [])]
    data = [Path(p) for p in out.get("data", [])]
    assert any("fig3_c2_coeff_compare" in p.name for p in pdfs)
    assert any("fig4_c3_cT_heatmap" in p.name for p in pdfs)
    # 檔案存在與 md5
    for p in pdfs + data:
        assert p.exists(), f"missing output: {p}"
        md5 = p.with_suffix(p.suffix + ".md5")
        assert md5.exists(), f"missing md5 for: {p}"
