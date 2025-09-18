# scripts/fig_nlo_offsets.py
# -*- coding: utf-8 -*-
import os, hashlib, argparse, math
import numpy as np
import matplotlib.pyplot as plt

from palatini_pt.gw.nlo import predict_offsets

DATA_DIR = "figs/data"
PDF_DIR  = "figs/pdf"

def _md5_write(path):
    with open(path, "rb") as f:
        h = hashlib.md5(f.read()).hexdigest()
    with open(path + ".md5", "w") as g:
        g.write(h + "\n")

def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

def build(config_path: str | None = None):
    # 預設參數（與論文文字一致，k^2/Λ^2）
    cfg = {
        "nlo": {
            "gradT_sq_eff": 1.0,           # a+b 中的 a
            "Ricci_deps_deps_eff": 0.3,    # b
            "Lambda2": 1.0e6,              # Λ^2
        }
    }
    # 可選：從 paper_grids.yaml 讀自訂參數（若沒裝 pyyaml 就跳過）
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r") as f:
                y = yaml.safe_load(f)
            if "nlo" in y: cfg["nlo"].update(y["nlo"].get("coeffs", {}))
            if "nlo" in y and "k" in y["nlo"]:
                kmin = float(y["nlo"]["k"].get("min", 1e-3))
                kmax = float(y["nlo"]["k"].get("max", 1e-1))
                npts = int(y["nlo"]["k"].get("n", 200))
            else:
                kmin, kmax, npts = 1e-3, 1e-1, 200
        except Exception:
            kmin, kmax, npts = 1e-3, 1e-1, 200
    else:
        kmin, kmax, npts = 1e-3, 1e-1, 200

    _ensure_dirs()
    k = np.logspace(math.log10(kmin), math.log10(kmax), npts)
    out = predict_offsets(k, cfg)

    # 存 CSV
    csv_path = os.path.join(DATA_DIR, "nlo_offsets.csv")
    with open(csv_path, "w") as f:
        f.write("k,delta_K,delta_G,delta_cT2\n")
        for i in range(len(k)):
            f.write(f"{k[i]},{out['delta_K'][i]},{out['delta_G'][i]},{out['delta_cT2'][i]}\n")
    _md5_write(csv_path)

    # 畫圖（log-log，顯示 slope 2）
    pdf_path = os.path.join(PDF_DIR, "fig8_nlo_offsets.pdf")
    plt.figure(figsize=(4.8, 3.2), dpi=180)
    plt.loglog(k, np.abs(out["delta_cT2"]), lw=1.6, label=r"$|\delta c_T^2(k)|$")
    plt.loglog(k, (out["delta_cT2"][0]/(k[0]**2))*k**2, ls="--", lw=1.0, label=r"$\propto k^2$")
    plt.xlabel(r"$k$"); plt.ylabel(r"$|\delta c_T^2|$")
    plt.title("NLO prediction: $\\delta c_T^2\\propto k^2/\\Lambda^2$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()
    _md5_write(pdf_path)

    print("Generated:")
    print(f"  [pdfs] {pdf_path}")
    print(f"  [data] {csv_path}")
    return [pdf_path, csv_path]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/paper_grids.yaml")
    args = ap.parse_args()
    build(args.config)

if __name__ == "__main__":
    main()
