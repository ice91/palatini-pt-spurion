# tests/test_cli.py
# -*- coding: utf-8 -*-
import subprocess
import sys
import shutil
import os
import pytest

def run_cmd(args, env=None):
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or os.environ.copy(),
    )
    return proc.returncode, proc.stdout, proc.stderr

@pytest.mark.skipif(shutil.which("palpt") is None, reason="palpt entrypoint not installed")
def test_cli_version():
    code, out, err = run_cmd(["palpt", "--version"])
    assert code == 0
    assert "palatini-pt-spurion" in out or "palpt" in out

@pytest.mark.skipif(shutil.which("palpt") is None, reason="palpt entrypoint not installed")
def test_cli_figs_smoke(tmp_path):
    # 在暫存目錄下執行，避免污染專案
    env = os.environ.copy()
    env["PALPT_OUTPUT_DIR"] = str(tmp_path)  # 若你的 CLI 支援環變更輸出
    code, out, err = run_cmd(["palpt", "figs", "--which", "smoke"], env=env)
    assert code == 0, f"stdout:\n{out}\nstderr:\n{err}"
