# %% [markdown]
# # 20 — Flux ratio R_{X/Y}(R) -> 1
# Finite-domain FRW ball visualization.

# %%
import numpy as np
import matplotlib.pyplot as plt
from palatini_pt.equivalence.flux_ratio import flux_ratio_FRW

# （可選）為了 Colab 更保險，明確指定使用 mathtext 而不是外部 TeX
import matplotlib as mpl
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.fontset"] = "stix"     # or 'dejavusans'
mpl.rcParams["font.family"] = "STIXGeneral"   # 視字型可用性而定

R = np.linspace(5.0, 80.0, 60)
out = flux_ratio_FRW(R, {"flux": {"sigma": 1.0, "c": 0.5}})
plt.plot(out["R"], out["R_DBI_CM"])
plt.axhline(1.0, ls="--")
plt.xlabel("R")
plt.ylabel(r"$\mathcal{R}_{X/Y}$")                 # <— 加上大括號
plt.title(r"Flux ratio → 1 with $R^{-\sigma}$ tail")  # <— 用 raw string
plt.tight_layout()
plt.show()
