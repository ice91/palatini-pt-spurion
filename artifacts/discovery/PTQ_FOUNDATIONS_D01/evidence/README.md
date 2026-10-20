# Reproducible new mathematics

Use the already installed SymPy 1.14.0 interpreter. From the project root:

```sh
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_FOUNDATIONS_D01/evidence/derive_EC_connection.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_FOUNDATIONS_D01/evidence/derive_models.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_FOUNDATIONS_D01/evidence/test_independent.py
```

Each script writes only its named JSON output beside itself. No script imports original project modules or accesses configured coefficients. `derive_models.py` differentiates the new action, solves the pairing relation, and constructs Hessian/Schur/response/equivalence expressions. `test_independent.py` does not import that script or the results JSON; it has a separate exterior algebra, reconstruction from diagonal control, source-transport calculation and counterexamples. `derive_EC_connection.py` builds the 24×24 exact connection-equation map from Levi-Civita and wedge signs.

Final executed results: 24 derivation checks, 12 independent tests, EC rank 24/determinant 16. The initial independent attempt counts (2 failures caused by noncanonical hyperbolic expressions) are preserved with a fix note. The repeated derivation execution incorporated the actual revised physical-action Hessian/source elimination, resolving an identified distinction rather than broadening the model search.

Theorems/assumptions and physical limits are in reports 01–07. No numerical fit or observational dataset is involved. “Independent” denotes an independent calculation path in the same sprint, not a separate human/laboratory validation or a claim of literature novelty.
