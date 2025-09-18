# scripts/fig_flux_ratio.py
# -*- coding: utf-8 -*-
import os, hashlib
import numpy as np
import matplotlib.pyplot as plt

# 依你的檔案結構（palatini_pt/equivalence/flux_ratio.py）
from palatini_pt.equivalence.flux_ratio import flux_ratio_FRW

DATA_DIR = "figs/data"
PDF_DIR  = "figs/pdf"
PDF_NAME = "fig9_flux_ratio.pdf"
CSV_NAME = "flux_ratio.csv"

def _md5_write(path: str) -> None:
    with open(path, "rb") as f:
        import hashlib
        h = hashlib.md5(f.read()).hexdigest()
    with open(path + ".md5", "w") as g:
        g.write(h + "\n")

def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

def _cfg_get(d, ks, default):
    cur = d if isinstance(d, dict) else {}
    for k in ks:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def build_from_config(config: dict | None):
    sigma = float(_cfg_get(config, ["flux","sigma"], 1.0))
    c     = float(_cfg_get(config, ["flux","c"], 0.5))
    Rmin  = float(_cfg_get(config, ["flux","R","min"], 5.0))
    Rmax  = float(_cfg_get(config, ["flux","R","max"], 80.0))
    npts  = int(_cfg_get(config,   ["flux","R","n"],   60))

    _ensure_dirs()
    R = np.linspace(Rmin, Rmax, npts)

    out = flux_ratio_FRW(R, {"flux": {"sigma": sigma, "c": c}})
    # out 應包含 keys: "R"、"R_DBI_CM"

    csv_path = os.path.join(DATA_DIR, CSV_NAME)
    with open(csv_path, "w") as f:
        f.write("R,R_DBI_CM\n")
        for i in range(len(out["R"])):
            f.write(f"{out['R'][i]},{out['R_DBI_CM'][i]}\n")
    _md5_write(csv_path)

    pdf_path = os.path.join(PDF_DIR, PDF_NAME)
    plt.figure(figsize=(4.8, 3.2), dpi=180)
    plt.plot(out["R"], out["R_DBI_CM"], lw=1.6, label=r"$\mathcal R_{X/Y}(R)$")
    plt.axhline(1.0, ls="--", lw=1.0)
    plt.xlabel(r"$R$")
    plt.ylabel(r"$\mathcal R_{X/Y}$")
    plt.title(r"Boundary flux ratio $\to 1$ with $R^{-\sigma}$ tail")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()
    _md5_write(pdf_path)

    return {"pdfs": [pdf_path], "data": [csv_path]}

def run(config: dict | None = None, which: str | None = None):
    """make_all_figs 的標準入口"""
    return build_from_config(config)

# 允許單獨執行：python -m scripts.fig_flux_ratio --config configs/paper_grids.yaml
def _load_yaml(path: str) -> dict | None:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paper_grids.yaml")
    args = ap.parse_args()
    cfg = _load_yaml(args.config)
    out = run(cfg)
    print("Generated:")
    for k, v in out.items():
        for x in v:
            print(f"  [{k}] {x}")
