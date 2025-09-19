# %% [markdown]
# # 20 — Flux ratio R_{X/Y}(R) -> 1
# Finite-domain FRW ball visualization.

# %%
import numpy as np, matplotlib.pyplot as plt
from palatini_pt.equivalence.flux_ratio import flux_ratio_FRW

R = np.linspace(5.0, 80.0, 60)
out = flux_ratio_FRW(R, {"flux": {"sigma": 1.0, "c": 0.5}})
plt.plot(out["R"], out["R_DBI_CM"])
plt.axhline(1.0, ls="--")
plt.xlabel("R"); plt.ylabel(r"$\mathcal R_{X/Y}$")
plt.title("Flux ratio → 1 with $R^{-\sigma}$ tail")
plt.show()
