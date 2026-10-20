# Reproduction and provenance — I04

Run from `/Users/rocky/Downloads/palatini-pt-spurion` using the already installed environment (Python 3.11, SymPy 1.14.0, mpmath 1.3.0):

```sh
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I04/evidence/derive_response.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I04/evidence/derive_feedback.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I04/evidence/check_independent.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I04/evidence/verify_preservation.py
```

The scripts write JSON only beside themselves. `-B` prevents bytecode outside the sandbox. No package was installed. The first three commands reproduce 15 response algebra checks, 7 geometric/feedback checks and 16 independent numerical cases respectively. They do not rerun I03 tests or count its heat kernel as an independent I04 experiment. Numeric dimensionless examples use m as a unit; Z and β choices are counterexamples/regularity illustrations, not physical parameter fits. Exact formulas and distributional/covariance proofs reside in the reports; tests alone do not prove gravity.

The final command verifies pre-existing regular files and generates final preservation/deliverable hashes; it intentionally excludes its generated hash manifest from hashing itself. It must be rerun if any deliverable changes. Running checks later may rewrite JSON, then the final command refreshes the new-file manifest. No command mutates Git or old research.

## Sources actually used

- `artifacts/discovery/PTQ_CLOUDS_I03/01_MICROSCOPIC_AREA_ENTROPY.md`: fixed scalar, state, signature, regulator and I03's limits.
- `artifacts/discovery/PTQ_CLOUDS_I03/02_INDUCED_GRAVITATIONAL_COUPLING.md`, Eqs.(16),(23),(29)–(32): local curvature/volume normalization and finite-counterterm obstruction.
- `artifacts/discovery/PTQ_CLOUDS_I03/03_FIRST_PRINCIPLES_GAP_AND_DECISION.md`, §§2–7: missing metric dynamics and proposed causal-response calculation. Its evidence files were inspected as provenance, not rerun as I04 experiments.
- I01 `03_INFORMATION_CAUSALITY_AND_TIME.md`, §§3–5, and I02 `03_ADVERSARIAL_AND_EXPLANATORY_VERDICT.md`, §§1–3: retarded-vs-advanced/state/observable discipline and coherent-state controls. Their 1+1D field is not substituted for I03's 3+1D massive parent.
- `refs/internal/INTERNAL_WHITEPAPER.md`: the constitutive/predictivity freeze remains closed; no verdict is rewritten.
- Primary external comparison read online: Anderson, Molina-París & Mottola, *Linear Response, Validity of Semi-Classical Gravity, and the Stability of Flat Space*, [author preprint gr-qc/0209075v1](https://arxiv.org/pdf/gr-qc/0209075), [published DOI](https://doi.org/10.1103/PhysRevD.67.024026). PDF physical pages 7–11 contain CTP/retarded/contact/subtraction formulas; pp.19–21 give Eq.(B37a) and subsequent dispersive integrals. Our channel is half their tensor-projector spectral coefficient. The PDF was read via the web tool and not copied into historical refs. No source-only audit or new historical Symmetry reconstruction was performed.

## Epistemic distinctions

The free scalar is an established parent, not a new matter hypothesis. The one-loop response is recomputed known QFT. Equation (22)'s dynamical metric, finite couplings, background stationarity and the semiclassical/EFT domain are explicit assumptions. The regular-branch bound, subtraction obstruction and gapped-parent no-go are analytic statements within stated domains. Quadratures independently verify coefficients/limits but provide no empirical or full nonlinear gravitational stability evidence. The exact higher-derivative runaway illustrates an uncontrolled resummation outside the order-reduced branch; it is not a prediction of a real low-energy instability.

The ordinary minimally coupled scalar has no gauge Gauss-law edge modes. No graviton/edge entropy or quantum metric constraint Hilbert space is included. None of OBP, Probability, Particles or PT imposes an extra equation on this calculation. No finite sharp-horizon entropy, Born-rule derivation, universal Clausius law, cosmological mechanism or black-hole information solution is inferred.
