# Reproduction and evidence boundaries

All scripts write only beside themselves. From repository root, use the installed interpreter:

```sh
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I02/evidence/derive_modular.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I02/evidence/derive_geometry_interface.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I02/evidence/calculate_detector_excess.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I02/evidence/test_independent.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I02/evidence/verify_preservation.py
```

Python3.11.13, SymPy1.14.0, mpmath1.3.0 were already installed; no dependency installation or production script is required. `-B` avoids incidental bytecode. The initial manifest is immutable evidence and must not be regenerated after the work.

## Actual calculations

| Script / result | What is actually checked | Limit |
|---|---|---|
| `derive_modular.py` / `modular_results.json` | 21 exact algebraic/symbolic checks: Lorentz/modular generators, actual wave/stress equation, coherent displacement and transformed Cauchy data, noncommuting regulator first-law/eigenvalues/Hessian, one-particle stress integrals; one separate compact bump quadrature check (22 total) | Does not reprove the existence of modular operators, type III or full BW theorem with computer tests |
| `derive_geometry_interface.py` / `geometry_results.json` | 9 exact checks: general conformal 2D Ricci/Einstein tensor, flat Rindler metric, coherent stress, conditional horizon integration/coefficient and null tensor algebra | Area entropy and universal Clausius are ADDED premises, not passed tests |
| `calculate_detector_excess.py` / `detector_results.json` | Direct finite-time amplitude integrals, separate vacuum response for the same switch, two durations; analytic unwindowed amplitude check | Perturbative response, not a full detector equilibration theory, experimental measurement or black-hole calculation |
| `test_independent.py` / `independent_results.json` | 10 unittest methods: independently differentiated eigenvalue entropy, matrix relative-entropy/dephasing/unitary examples, numerical derivative/symplectic compact-packet integrals, independent geometric tensor contraction/null-direction solve, frequency-convolution vs time-integral detector check, local nonprojective POVM, momentum/position one-particle energy | No imports of derivation scripts, prior project code or result JSON; tests verify concrete formulas, not entire continuum physics by counting successes |

The relative-entropy positivity proof, coherent Weyl-cocycle calculation, first-law differentiation, local-instrument impossibility proof and conditional geometric argument are displayed analytically in the reports. They are distinguished from numerical illustrations and assumed existing theorems. No scientific hypothesis is accepted merely because scripts agree.

The modular script initially had 19 checks; a redundant placeholder identity was replaced by two actual transformed-Cauchy-data differentiations, giving 20. Adding the same I01 one-particle stress/energy dictionary gave 22, with independent test methods increasing from 9 to 10. All executed mathematical runs passed. This was a bounded calculation extension, not a new parent/revision or 41 independent experiments. No failed mathematical run is hidden. The substantive cycles were the modular/coherent/first-law calculation and its adversarial geometric/one-particle/operational checks.

## Source and epistemic pointers

Local references are hashed in `initial_state.json`, along with all pre-existing project files:

- I01's four reports and actual `derive_observer.py`, `test_independent.py`, result files: fixed current parent, type-III caveat, detector and one-particle amplitude. Prior tests were not rerun/count-promoted as new independent experiments.
- D02's three reports and `derive_parent_geometry.py`, `derive_fixed_parent_projections.py`, results: sourced torsion and matched-source equivalence in a distinct supplied EH parent, not gravity derived by information.
- `refs/preprints202604.1054.v3.pdf`: OBP §§2–5.6, conditional organizing principles and separate probability structure.
- `refs/preprints202604.0888.v2.pdf`: Probability §§2–4, conditional operational-equivalence framework, not a gravitational theorem.
- Exact external Particles9,93: `/Users/rocky/Downloads/particles-09-00093.pdf`, pp.1–2/Eqs.(1)–(2), DOI10.3390/particles9030093; exact SHA256 is in the initial manifest. No unrelated manuscript was substituted.
- `refs/internal/INTERNAL_WHITEPAPER.md`, §§1,13–18: freeze/reopen contract, read before the new construction. No constitutive dictionary or reopening was attempted.

Primary author texts consulted read-only online, no external archive copied:

- Witten2018, *Notes on Some Entanglement Properties of Quantum Field Theory*, §3.3 Eqs.(3.28)–(3.29), §§5–6: https://arxiv.org/html/1803.04993v6 . Established relative-modular construction, positivity and continuum boundary.
- Casini–Grillo–Pontello2019, *Relative entropy for coherent states from Araki formula*, §5.2.1 Eqs.(5.11)–(5.13), Eq.(5.8): https://arxiv.org/html/1903.00109 (displayed v3,2019-07-04). Established inner-unitary method; our current-domain cocycle/stress normalization is recomputed rather than importing an unrestricted massless scalar.
- Jacobson1995, *Thermodynamics of Spacetime: The Einstein Equation of State*, Eqs.(1)–(6): https://arxiv.org/html/gr-qc/9504004 (displayed v2,1995-06-06). Known conditional higher-dimensional template, not a new QFT parent.
- Blanco–Casini–Hung–Myers2013, *Relative Entropy and Holography*, https://arxiv.org/abs/1305.3182 was checked for bibliographic first-law context. Its requested HTML endpoint failed; no inaccessible body content or holographic assumption was used. Our finite first-law derivation is explicit in report01.

Search results from nonprimary websites were not relied on. None of the historical Symmetry audits was redone or altered. No external communication, publishing, Git mutation, installation or fitting occurred.

## Preservation

`initial_state.json` contains 1160 pre-existing regular project-file hashes, original HEAD/main/status/refs/untracked inventory, and one external exact PDF hash. The verifier checks each original, additions outside I02, Git state/diffs and final result integrity. `deliverable_sha256.json` covers every sandbox file except itself, including the final preservation report. The manifest cannot contain its own hash without circularity. File inventory excludes `.git` internals; HEAD/branch/refs and tracked/staged diffs are separately checked. Freeze remains CLOSED; STOP after this sprint.
