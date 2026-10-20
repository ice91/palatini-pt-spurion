# scripts/fig_flux_ratio.py
# -*- coding: utf-8 -*-
import os, hashlib, argparse, numpy as np, matplotlib.pyplot as plt

from palatini_pt.equivalence import flux_ratio_FRW

DATA_DIR = "figs/data"
PDF_DIR  = "figs/pdf"

def _md5_write(path):
    import hashlib
    with open(path, "rb") as f:
        h = hashlib.md5(f.read()).hexdigest()
    with open(path + ".md5", "w") as g:
        g.write(h + "\n")

def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)

def build(config_path: str | None = None):
    # 預設掃描範圍
    Rmin, Rmax, npts = 5.0, 80.0, 60
    sigma, c = 1.0, 0.5

    # 可選：從 YAML 覆寫
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r") as f:
                y = yaml.safe_load(f)
            if "flux" in y:
                sigma = float(y["flux"].get("sigma", sigma))
                c     = float(y["flux"].get("c", c))
                if "R" in y["flux"]:
                    Rmin = float(y["flux"]["R"].get("min", Rmin))
                    Rmax = float(y["flux"]["R"].get("max", Rmax))
                    npts = int(y["flux"]["R"].get("n", npts))
        except Exception:
            pass

    _ensure_dirs()
    R = np.linspace(Rmin, Rmax, npts)
    out = flux_ratio_FRW(R, {"flux": {"sigma": sigma, "c": c}})

    # 存 CSV
    csv_path = os.path.join(DATA_DIR, "flux_ratio.csv")
    with open(csv_path, "w") as f:
        f.write("R,R_DBI_CM\n")
        for i in range(len(out["R"])):
            f.write(f"{out['R'][i]},{out['R_DBI_CM'][i]}\n")
    _md5_write(csv_path)

    # 畫圖
    pdf_path = os.path.join(PDF_DIR, "fig9_flux_ratio.pdf")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4.8, 3.2), dpi=180)
    plt.plot(out["R"], out["R_DBI_CM"], lw=1.6, label=r"$\mathcal R_{X/Y}(R)$")
    plt.axhline(1.0, ls="--", lw=1.0)
    plt.xlabel(r"$R$"); plt.ylabel(r"$\mathcal R_{X/Y}$")
    plt.title(r"Boundary flux ratio $\to 1$ with $R^{-\sigma}$ tail")
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
