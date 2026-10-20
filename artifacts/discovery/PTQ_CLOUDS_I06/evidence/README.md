# I06 reproduction and epistemic limits

Run from `/Users/rocky/Downloads/palatini-pt-spurion` in the already installed Python 3.11 / SymPy 1.14.0 environment:

```sh
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I06/evidence/derive_geometry.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I06/evidence/derive_dynamics.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I06/evidence/check_independent.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I06/evidence/verify_preservation.py
```

Only new I06 JSON files are written. `-B` suppresses bytecode. No installation, original-source execution or original-file mutation is needed. Rerun preservation after changing a new deliverable; its own output hash manifest is excluded from self-hashing.

## Executed calculations

- `geometry_results.json`: **21** exact checks. A single rank-one trace-square deformation; Gram invariants and relaxed constraints; exact trace inversion/potential; constraint/tangent ranks; B-induced Maxwell Hodge; all-24-component auxiliary connection inversion; derived scalar kinetic density for four arbitrary gradient components; the alternate topological coordinate chart of the same family.
- `dynamics_results.json`: **20** exact checks. Canonical scalar redefinition/mass; actual beta Euler derivative; nonzero nonlinear momentum/Hessian; full seven-velocity Legendre rank and four-constraint linear matrix; scalar Hamiltonian bracket; Maxwell symbol/gauge ranks and kinetic sign; exact nonzero-beta Cauchy constraint witness and hard-retraction failure. This does not substitute numerical rank tests for the generic analytic rank proof.
- `independent_results.json`: **18** independent rational cases. A separately implemented exterior-action Hessian (not the DB equation and not imported from either script) is solved at three rational beta values with timelike, spacelike and mixed gradients. A metric Hodge formula at a nondiagonal coframe independently matches the B conjugation map. Invariants, exact initial constraints and different scalar pole ratios are checked. No numerical/empirical data or fit is used.

Two substantive cycles, one BF family, one controlled relaxation. The alternate gamma chart is the same constrained B family, not a new candidate. No model revision, full-rank twenty-channel penalty, Pauli–Fierz tuning or new chiral model is introduced. `execution_notes.md` records bookkeeping/lexical corrections. All final checks pass.

## Inputs read and source calibration

- All three `artifacts/discovery/PTQ_CLOUDS_I05/*.md` reports and its actual simplicity derivation script were read. Their hard branch/mode result is a control, not relabeled as a new I06 experiment.
- `refs/internal/INTERNAL_WHITEPAPER.md`: IB-2/constitutive predictivity and fitting stay closed.
- I01–I04's established distinctions between accessible states, probability pairing, retarded support, modular data and a physical metric law are retained; none is promoted to a simplicity proof. Their original state/loop calculations were not rerun.
- Alexandrov–Krasnov, [0809.4763](https://arxiv.org/pdf/0809.4763), §3, especially PDF pp.8–9: non-chiral generalized canonical count depends on multiplier-potential Hessian rank. Its generic full-rank eight-mode count is not applied to our rank-one deformation.
- Krasnov, [0811.3147](https://arxiv.org/pdf/0811.3147), Introduction and §§II–IV, Eqs.(2)–(8): modified chiral SO(3) framework and two-mode statement; complex/reality domain is separate from our real non-chiral one. This is a comparison source, not an independently verified healthy-real counterexample in this model.
- Taveras–Yunes, [0807.2652v2](https://arxiv.org/pdf/0807.2652), Eq.(49), PDF p.7: primary benchmark for the varying-Holst scalar kinetic term. The potential and operational map in I06 are calculated from our defined action; no cosmological conclusions of that paper are used.
- Choquet-Bruhat–Isenberg–Pollack, [gr-qc/0611009](https://arxiv.org/pdf/gr-qc/0611009), §§2–4: Einstein–scalar equations, constraints and local well-posedness. We calculate exact admissible initial data and invoke the known local theorem on the regular smooth-potential patch; no fresh global/PDE theorem is claimed. Potential walls and singular strata are not covered by a claimed global stability result.

Primary references were consulted read-only via web tools. No PDF archive was duplicated and nothing was added to historical refs. Known benchmarks are not new physical discoveries.

## Boundaries of the conclusions

Positive local kinetic terms, flat-vacuum mass stability and local hyperbolic development do not prove complete nonlinear/global or quantum/UV health. The selected Lorentz/volume/sign component, nineteen remaining algebraic constraints, the BF action and both probe prescriptions are independently supplied inputs. Only the Maxwell constitutive map is directly computed from B; a clock using the same normalized metric is an additional assumed common-metric rule.

Nonlinear regular ranks are justified analytically by surjectivity of the invertible-B Gram derivative, exact coframe/scalar reconstruction, invertible connection elimination and a nonzero scalar Legendre block. The original singular first-order Dirac matrix is not solved on the topological/degenerate strata; no generic count is extrapolated there. The counterexample's Cauchy data are a local existence witness, not another perturbation background, a cosmological solution study or a fit.

The idempotent algebraic hard retraction is not a map of generic physical solutions. Observer restriction, gauge equivalence, Hilbert projection and Schur elimination remain different operations. The sprint derives no microscopic origin of the assumed constraints, no universal matter cone, no functional IB-2 closure and no retroactive proof of Symmetry.
