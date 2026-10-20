# %% [markdown]
# # 00 — sanity
# Quick import + TT luminality smoke (should be exactly 1 at LO).

# %%
import numpy as np
from palatini_pt.gw.quadratic_action import tensor_action_coeffs
from palatini_pt.spurion.pt_even import project_scalar
from palatini_pt.spurion.profile import constant_gradient

# %%
k = np.logspace(-3, -1, 64)
deps = constant_gradient([0.0, 0.0, 1.0])  # ∂ε 指向 z
Seps = project_scalar(np.dot(deps, deps))

K, G = tensor_action_coeffs(Seps)
cT2 = G / K
print("K(LO)=", K, "G(LO)=", G, "c_T^2(LO)=", cT2)
assert np.allclose(cT2, 1.0), "locked LO must be luminal"
