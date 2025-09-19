# scripts/fig_nlo_offsets.py
# -*- coding: utf-8 -*-
"""
Fig.8 — NLO offsets: δc_T^2(k) ~ b * k^2 / Λ^2
- 讀取 configs/* 中的 nlo 區塊
- 產生 CSV: figs/data/nlo_offsets.csv
- 產生 PDF: figs/pdf/fig8_nlo_offsets.pdf
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "figs/data"
PDF_DIR  = "figs/pdf"
PDF_NAME = "fig8_nlo_offsets.pdf"
CSV_NAME = "nlo_offsets.csv"

def _md5_write(path: str) -> None:
    try:
        import hashlib
        with open(path, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
        with open(path + ".md5", "w") as g:
            g.write(h + "\n")
    except Exception:
        pass

def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PDF_DIR,  exist_ok=True)

def _cfg_get(d, ks, default):
    cur = d if isinstance(d, dict) else {}
    for k in ks:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _compute_nlo_offsets_from_config(cfg: dict | None) -> dict:
    # k-grid
    kmin = float(_cfg_get(cfg, ["nlo", "k", "min"], 1.0e-3))
    kmax = float(_cfg_get(cfg, ["nlo", "k", "max"], 1.0e-1))
    npts = int(_cfg_get(cfg, ["nlo", "k", "n"],   200))
    k = np.geomspace(kmin, kmax, npts)

    # 係數（對應論文：δc_T^2 ≈ b * k^2 / Λ^2）
    Lambda2 = float(_cfg_get(cfg, ["nlo", "coeffs", "Lambda2"], 1.0e6))
    b       = float(_cfg_get(cfg, ["nlo", "coeffs", "Ricci_deps_deps_eff"], 0.0))

    # 允許有 a 但對 δc_T^2 僅 b 會進入（a 只會同時改變 K, G，不改差值）
    # a = float(_cfg_get(cfg, ["nlo", "coeffs", "gradT_sq_eff"], 0.0))

    # 主結果：可能為 0（例如 b=0 或 Lambda 很大）
    delta_cT2 = (b * (k**2)) / (Lambda2 if Lambda2 != 0.0 else 1.0e99)

    return {"k": k, "delta_cT2": delta_cT2}

def _save_csv(out: dict) -> str:
    csv_path = os.path.join(DATA_DIR, CSV_NAME)
    with open(csv_path, "w") as f:
        f.write("k,delta_cT2\n")
        for ki, yi in zip(out["k"], out["delta_cT2"]):
            f.write(f"{ki},{yi}\n")
    _md5_write(csv_path)
    return csv_path

def _plot_pdf(out: dict) -> str:
    pdf_path = os.path.join(PDF_DIR, PDF_NAME)
    k   = np.asarray(out["k"], dtype=float)
    y   = np.asarray(out["delta_cT2"], dtype=float)
    yabs = np.abs(y)
    pos  = yabs > 0

    plt.figure(figsize=(4.8, 3.2), dpi=180)

    if pos.sum() >= 2:
        # 有足夠正值：log–log 繪圖 + 參考斜率 k^2
        kpos = k[pos]
        ypos = yabs[pos]
        plt.loglog(kpos, ypos, lw=1.6, label=r"$|\delta c_T^2(k)|$")

        # 參考 k^2：用第一點做歸一
        ref = (ypos[0] / (kpos[0]**2)) * (kpos**2)
        plt.loglog(kpos, ref, ls="--", lw=1.0, label=r"$\propto k^2$")

        plt.ylabel(r"$|\delta c_T^2|$")
        # ---- 斜率擬合（log–log）與 Λ 估計 ----
        try:
            X = np.log(kpos)
            Y = np.log(ypos)
            m, b = np.polyfit(X, Y, 1)   # Y ≈ m X + b
            # 從 δcT^2 = bcoef * k^2 / Λ^2 推回 Λ^2 的 robust 估計
            # 需要配置裡的 bcoef（名為 Ricci_deps_deps_eff）
            bcoef = float(_cfg_get(globals().get("CFG_CACHE", None), ["nlo", "coeffs", "Ricci_deps_deps_eff"], 0.0))
            if bcoef > 0:
                lam2_est = np.median(bcoef*(kpos**2)/ypos)
                lam_est  = np.sqrt(lam2_est)
                txt = rf"slope≈{m:.2f},  $\hat\Lambda$≈{lam_est:.2e}"
            else:
                txt = rf"slope≈{m:.2f}"
            plt.text(0.03, 0.05, txt, transform=plt.gca().transAxes, fontsize=9)
        except Exception:
            pass
    else:
        # 全為 0 或只有 1 個正值：避免 log-scaling 警告，改線性 y 軸
        plt.plot(k, yabs, lw=1.6, label=r"$|\delta c_T^2(k)|\approx 0$")
        plt.xscale("log")
        plt.ylabel(r"$|\delta c_T^2|$ (linear)")

    plt.xlabel(r"$k$")
    plt.title(r"NLO offsets: $\delta c_T^2 \propto k^2/\Lambda^2$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()
    _md5_write(pdf_path)
    return pdf_path

def build_from_config(config: dict | None) -> dict:
    _ensure_dirs()
    out = _compute_nlo_offsets_from_config(config)
    csv_path = _save_csv(out)
    pdf_path = _plot_pdf(out)
    return {"pdfs": [pdf_path], "data": [csv_path]}

def run(config: dict | None = None, which: str | None = None) -> dict:
    """供 make_all_figs 呼叫的標準入口"""
    return build_from_config(config)

# 允許單檔執行：python -m scripts.fig_nlo_offsets --config configs/paper_grids.yaml
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
