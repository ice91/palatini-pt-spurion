# I03 — reproducibility and evidence boundaries

Run from the project root with the already installed interpreter:

```sh
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I03/evidence/derive_area_entropy.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I03/evidence/derive_induced_curvature.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I03/evidence/test_independent.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I03/evidence/verify_preservation.py
```

Python3.11/SymPy1.14/mpmath1.3 suffice. No installation, network retrieval, original project module, production script or configured coefficient is needed to reproduce the mathematics. Each script writes only inside this evidence directory; `-B` prevents incidental bytecode. The initial manifest is immutable evidence, not something to regenerate after execution.

## Two main calculations

- `derive_area_entropy.py` / `area_results.json`: 14 exact algebraic/symbolic checks. It obtains the cone constant from an angular geometric sum and the bulk-subtracted radial Abel limit, performs the transverse momentum integral, differentiates the normalized replica functional, checks discrete orbifold images and retains the finite-Rényi/EH extrapolation counterexample. It does not insert A/(4G) as the target construction.
- `derive_induced_curvature.py` / `curvature_results.json`: 18 exact checks. It contracts the Riemann-normal-coordinate Van Vleck expansion, extracts its curvature coefficient into the determinant integral, computes the massive integral/sign, differentiates the smooth-cap geometry, compares the entropy, tracks the volume term, shared smooth-profile control, species scaling and arbitrary bare normalization. It does not postulate metric stationarity.
- `test_independent.py` / `independent_results.json`: 11 unittest methods, no imports of either derivation script, output JSON or previous project code. Direct numerical Bessel Laplace transforms, image radial integrations, independent S² eigenvalue sums/Euler–Maclaurin, massive proper-time quadrature, smooth-profile integral, smooth-cap integral, Gaussian normalized-source counterterm identity, replica/Rényi and species/state/volume controls. Precision50 digits; quoted asymptotic/extrapolation errors have their own tolerances.

The continuum replica construction, geometric transport identities, Gaussian field measure and heat-kernel local-expansion domain are stated assumptions/standard mathematical input. Analytic derivations are displayed in the reports. No test count is treated as proof of all continuum physics or a gravitational consistency gate. The numerical masses/cutoffs in tests are dimensionless verification samples, not fitted parameters or a chosen physical Newton constant. Auxiliary S²×R² is a check of the same scalar heat kernel on an external Euclidean geometry, not a second microscopic field/vacuum model.

`area_attempt1.json` records the one initial failure: SymPy did not reduce coth(u/2)/(2sinh u) to 1/[2(cosh u−1)] using its default simplifier. The assertion failed on the unreduced expression. Rewriting into exponentials verified exact equality; no expected coefficient, action, regulator or domain was changed. Subsequent area derivation and all independent tests passed. No physical model revision or nonminimal action was implemented.

The counterfactual nonminimal-coupling discussion in report03 is explicitly established scope information, not an executed second model. Changing the proper-time profile is a regulator-dependence check on the same scalar determinant, not a second physical UV theory.

## Sources and actual usage

Read-only project starting evidence:

- I01's parent/observer-reduction/information/time/verdict reports and its actual derivative/counterstate code; the sharp-cut/type-III and operation distinctions are retained.
- I02's three reports, especially its G3/G5 premise table, coherent-state controls and actual modular derivation/results. Its conditional Einstein argument is not counted as a new calculation here.
- `refs/internal/INTERNAL_WHITEPAPER.md`: status freeze and §§1,13–18. Constitutive IB-2 and empirical work remain CLOSED. No source/manuscript/audit modification.
- OBP/Probability/Particles are not mathematical inputs to the cone/heat-kernel computation. Their organizing/probability/block-reduction ideas cannot supply its coefficients or a metric variational principle. No unrelated manuscript replaces an exact reference.

Primary author texts consulted read-only online:

- Solodukhin, *Entanglement entropy of black holes*, Living Rev.Rel.14,8 (2011), §§2.8–2.10 Eqs.(20),(27)–(31), §3.16 Eqs.(141)–(144): https://arxiv.org/html/1104.3712 . Benchmark normalization and scope warning for nonminimal/contact terms; no black-hole model is imported.
- Vassilevich, *Heat kernel expansion: user's manual*, Phys.Rept.388,279 (2003), Eq.(4.27): https://arxiv.org/html/hep-th/0306138 . Standard geometric coefficient; its a_2 convention corresponds to the coefficient of s denoted a_1 here.
- Susskind–Uglum, Phys.Rev.D50,2700 (1994): https://doi.org/10.1103/PhysRevD.50.2700 and https://arxiv.org/abs/hep-th/9401070 . Historical benchmark for entropy/coupling renormalization, not a source of a new scalar action or a proof of quantum gravity here.
- NIST DLMF §§10.22,10.43 were checked for special-function context. The Bessel normalization/Laplace identities actually used are written in report01 and independently numerically checked; no uninspected formula number is claimed.

A guessed `hep-th/9505120` link resolved to an unrelated orbifold paper; none of its contents was used or substituted as evidence. Nonprimary search results were not relied on. No large PDF archive or previous audit tree was copied.

## Preservation

`initial_state.json` inventories 1176 pre-existing regular project files excluding `.git` internals, records HEAD/main/status/refs/untracked material and SHA256 hashes. `verify_preservation.py` compares every byte hash, additions outside I03, Git refs/state and tracked/staged diffs; it also checks final math-result integrity. `deliverable_sha256.json` covers all final I03 files except itself, including the preservation report; a manifest cannot hash itself without circularity.

No original files or Git refs/history were mutated. No dependency installation, empirical fitting, publication or external communication occurred. Read-only online source consultation is distinct from contacting authors/journals. STOP after the bounded sprint; no automatic next calculation.
