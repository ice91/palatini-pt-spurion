# PTQ_FOUNDATIONS_D01 — research charter

**This is NEW research, not retroactive proof of Symmetry.** Historical G0/T0, original contradictions, the internal whitepaper and constitutive-predictivity freeze are unchanged. The present user authorization permits a separate bounded model construction, actual variation, symbolic checks and falsification. It does not reopen IB-2, SPARC/cosmology fitting, H_ref, C_PTQ, arbitrary constitutive generators or canonical predictivity claims.

## Epistemic labels

- **ESTABLISHED EXISTING RESULT:** Cartan calculus, covariant change of variables, finite-dimensional self-adjointness/similarity and Schur block elimination; historical audits retain their original domain.
- **NEW CANDIDATE DEFINITION / NEW ASSUMPTION:** every new domain, action, PT lift, carrier, constant parameter, pairing, state/source prescription below. These are this sprint's choices, never attributed to the original paper.
- **NEW CONDITIONAL MATHEMATICAL RESULT:** the displayed lemmas and action-derived formulas, conditional on those choices. “New” means newly derived/tested here, not a claim of literature priority.
- **NEW UNVERIFIED PHYSICAL HYPOTHESIS:** any interpretation as a microscopic universe, quantum gravity, empirical PT effect, or uniquely physical metric. No such hypothesis is promoted to a result.

## Bounded search and actual work

Exactly two candidates and two substantial cycles:

1. Real Riemann–Cartan Einstein–Cartan seed with a natural anti-linear pullback. Verify the lift and derive a limited covariance no-go; derive its vacuum connection equation rather than impose pure trace.
2. A minimally enriched constant-coefficient scalar doublet on the first candidate's flat vacuum, with one internal grading and anti-linear PT. Derive the action, equations, physical pairing criterion, Hessians, Schur operators, pole response and counterexamples.

Exactly **one controlled model revision**: the initial independent-dual carrier action supplies equations but does not supply a positive physical pairing. A negative bare-coordinate pole residue falsifies its naive physical interpretation. The revision uses a solved positive pairing and real physical action with the same equations, and transports sources/detectors consistently. It changes a new model's physical interpretation, never an old source. Subsequent source dressing is part of this same revision. A test simplification fix is recorded separately and is not another model revision.

No third candidate, interaction scan, derivative-mixing model, field-dependent constitutive function, ADM/Hamiltonian/PDE research program or observational fit was launched. The scalar sector's exact constant-coefficient wave well-posedness is proved by reduction to ordinary Klein–Gordon equations, not by a new nonlinear PDE analysis.

## Evidence and reproducibility

Working directory: `/Users/rocky/Downloads/palatini-pt-spurion`.
Initial HEAD: `6b456b89baa866b6dd05bc69a837b98a985a34ad`; branch `main`.
Initial grouped status: untracked `.DS_Store`, `artifacts/`, `refs/`.
The new sandbox did not exist; no prior result was overwritten.

`evidence/initial_state.json` records all 1,108 pre-existing regular project-file SHA256 values, exact untracked paths, Git refs/HEAD/status and the exact local Particles article hash. New output is confined to `artifacts/discovery/PTQ_FOUNDATIONS_D01/`.

Installed interpreter `/Users/rocky/miniconda3/envs/palpt/bin/python`, SymPy 1.14.0. No dependency installed. Reproduce from project root:

```sh
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_FOUNDATIONS_D01/evidence/derive_EC_connection.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_FOUNDATIONS_D01/evidence/derive_models.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_FOUNDATIONS_D01/evidence/test_independent.py
```

The derivation script differentiates actions and constructs matrices; the independent test script does not import it or its output. It reconstructs the model from an ordinary diagonal control, computes source weights independently, checks actual physical-action elimination, verifies Cartan/Hodge properties in a separate exterior algebra, and supplies counterexamples. Symbolic theorem proofs supplement tests; test success alone is not physical evidence.

Evidence: `EC_connection_results.json` (24×24 connection equation rank 24, determinant 16); `derivation_results.json` (24 exact checks); `independent_test_results.json` (12 tests, all pass). Initial independent test counts and normalization-fix notes are retained: two structural-equality failures arose from unsimplified hyperbolic half-angle expressions and were corrected by exponential normalization without changing a model or expected identity.

## Reference boundary

Symmetry `refs/symmetry-18-00170-v2.pdf`, original audits `artifacts/audits/SYM_FULL_A01/`, `artifacts/audits/SYM_HIST_P01/`, and completed G01/T01 gates are immutable historical evidence. Particles 9,93 is available exactly at `/Users/rocky/Downloads/particles-09-00093.pdf`, SHA256 `dd41e88ebd180f9b73312da95d981280aacc0166ccfb515bd0054785b4ba54f2`. Its pp.4–10 distinguish grading, PT, inverse prescription and Schur; this sprint defines a different scalar mass-mixing model, not its geometric principal-symbol deformation. Probability `refs/preprints202604.0888.v2.pdf`, PDF pp.6–8, motivates complete source/measurement transport. `refs/manuscript.pdf` is a distinct auxiliary model and is not used to supply new gravity dynamics. No network action, Git mutation, manuscript revision or publication occurred.
