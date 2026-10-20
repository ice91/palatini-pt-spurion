# %% [markdown]
# # 10 — C2 symbolic demo
# Show DBI / closed-metric reduce to A_* * I_T at quadratic order.

# %%
import sympy as sp
from palatini_pt.equivalence.coeff_extractor import reduce_to_IT

lam, eta, Seps = sp.symbols("lambda eta Seps", positive=True)
res_dbi = reduce_to_IT(route="dbi", lam=lam, eta=eta, Seps=Seps)
res_cm  = reduce_to_IT(route="closed_metric", lam=lam, eta=eta, Seps=Seps)
sp.simplify(res_dbi - res_cm)
