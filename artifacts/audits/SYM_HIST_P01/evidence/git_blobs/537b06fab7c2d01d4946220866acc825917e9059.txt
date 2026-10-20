# scripts/fig_nlo_offsets.py
# -*- coding: utf-8 -*-
import os, hashlib, math
import numpy as np
import matplotlib.pyplot as plt

# 依照 tests/test_nlo.py 的慣例，這個 API 名稱通常存在
from palatini_pt.gw.nlo import predict_offsets

DATA_DIR = "figs/data"
PDF_DIR  = "figs/pdf"
PDF_NAME = "fig8_nlo_offsets.pdf"
CSV_NAME = "nlo_offsets.csv"

def _md5_write(path: str) -> None:
    with open(path, "rb") as f:
        h = hashlib.md5(f.read()).hexdigest()
    with open(path + ".md5", "w") as g:
        g.write(h + "\n")

def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

def _cfg_get(d, ks, default):
    """安全取 config dict：_cfg_get(cfg, ['nlo','coeffs','Lambda2'], 1e6)"""
    cur = d if isinstance(d, dict) else {}
    for k in ks:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def build_from_config(config: dict | None):
    # 讀參數（有就用，沒有就用預設）
    a   = float(_cfg_get(config, ["nlo","coeffs","gradT_sq_eff"], 1.0))
    b   = float(_cfg_get(config, ["nlo","coeffs","Ricci_deps_deps_eff"], 0.3))
    L2  = float(_cfg_get(config, ["nlo","coeffs","Lambda2"], 1.0e6))

    kmin = float(_cfg_get(config, ["nlo","k","min"], 1e-3))
    kmax = float(_cfg_get(config, ["nlo","k","max"], 1e-1))
    npts = int(_cfg_get(config,   ["nlo","k","n"],   200))

    _ensure_dirs()
    k = np.logspace(math.log10(kmin), math.log10(kmax), npts)

    # predict_offsets 介面：由你的 tests 可知會吃 (k, cfg)
    out = predict_offsets(k, {
        "nlo": {
            "coeffs": {
                "gradT_sq_eff": a,
                "Ricci_deps_deps_eff": b,
                "Lambda2": L2,
            }
        }
    })

    # 存 CSV
    csv_path = os.path.join(DATA_DIR, CSV_NAME)
    with open(csv_path, "w") as f:
        f.write("k,delta_K,delta_G,delta_cT2\n")
        for i in range(len(k)):
            f.write(f"{k[i]},{out['delta_K'][i]},{out['delta_G'][i]},{out['delta_cT2'][i]}\n")
    _md5_write(csv_path)

    # 畫圖
    pdf_path = os.path.join(PDF_DIR, PDF_NAME)
    plt.figure(figsize=(4.8, 3.2), dpi=180)
    plt.loglog(k, np.abs(out["delta_cT2"]), lw=1.6, label=r"$|\delta c_T^2(k)|$")
    # 參考斜率 ~ k^2
    ref = (abs(out["delta_cT2"][0])/(k[0]**2))*k**2 if k[0] > 0 else k**2
    plt.loglog(k, ref, ls="--", lw=1.0, label=r"$\propto k^2$")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$|\delta c_T^2|$")
    plt.title(r"NLO: $\delta c_T^2 \propto k^2/\Lambda^2$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()
    _md5_write(pdf_path)

    return {"pdfs": [pdf_path], "data": [csv_path]}

def run(config: dict | None = None, which: str | None = None):
    """make_all_figs 的標準入口"""
    return build_from_config(config)

# 允許單獨執行：python -m scripts.fig_nlo_offsets --config configs/paper_grids.yaml
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
