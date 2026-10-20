# D02 reproducible calculations

One fixed new geometric parent, two physically motivated retained lines, zero model revisions. All code/output lives here and imports no original project implementation or configured coefficients.

From project root, using installed SymPy 1.14.0:

```sh
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_FOUNDATIONS_D02/evidence/derive_parent_geometry.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_FOUNDATIONS_D02/evidence/derive_fixed_parent_projections.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_FOUNDATIONS_D02/evidence/test_independent_geometry_and_response.py
```

`derive_parent_geometry.py`: unconstrained 24-component compatible contorsion; builds curvature, differentiates the true Hessian, solves the sourced system, tests the pure-trace result and exact auxiliary Schur block. 9 exact checks; Hessian rank 24, determinant 256 in declared coordinates.

`derive_fixed_parent_projections.py`: constructs action-derived K, both K-orthogonal projectors, actual Hessian charts, closure commutators, full and Schur responses with the same source/detector and all-source matrix identities. 19 exact checks.

`test_independent_geometry_and_response.py`: imports neither derivation script nor its result. Independent trace completion plus all 24 curvature variations via commutators, independent scalar-jet action variation, wedge/NY versus norm, matched-source solves, source contact, closure and singular-locus tests. 11 tests, all pass. Initial factored-expression equality failure and output-formatting fix are recorded; no physical target/parameter changed.

JSON files record exact expressions and pass/fail, not hardcoded experimental data. Analytic proofs and physical assumptions are in reports 01–03. There are no fitted numbers, numerical illustrations counted as evidence, or claims of a separate laboratory/human validation. Final preservation covers all pre-existing project files and recorded Git state.
