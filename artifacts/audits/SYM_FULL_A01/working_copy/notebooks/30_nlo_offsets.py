# %% [markdown]
# # 30 — NLO offsets
# δc_T^2(k) ~ b k^2 / Λ^2

# %%
import numpy as np, matplotlib.pyplot as plt
from palatini_pt.gw.nlo import predict_offsets

k = np.logspace(-3, -1, 200)
cfg = {"nlo": {"gradT_sq_eff": 1.0, "Ricci_deps_deps_eff": 0.3, "Lambda2": 1e6}}
out = predict_offsets(k, cfg)

plt.loglog(k, np.abs(out["delta_cT2"]), label=r"$|\delta c_T^2|$")
plt.loglog(k, (out["delta_cT2"][0]/(k[0]**2))*k**2, ls="--", label=r"$\propto k^2$")
plt.legend(); plt.xlabel("k"); plt.ylabel(r"$|\delta c_T^2|$")
plt.title("NLO: $k^2/\\Lambda^2$")
plt.show()
