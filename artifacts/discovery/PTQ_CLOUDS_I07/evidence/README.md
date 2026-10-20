# I07 reproduction and source discipline

This is a bounded classical closure calculation, not a quantum path integral or a particle spectrum computation. One parent action, one coarse map, two cycles (derivation and independent verification). The C19 early-stop condition was reached. No new mechanism or effective penalty is supplied.

## Reproduction

From the repository root, using the already installed environment:

```sh
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I07/evidence/derive_closure.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I07/evidence/check_independent.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I07/evidence/verify_preservation.py
```

The first command produces `closure_results.json` (30 checks); the second produces `independent_results.json` (21 checks). The second uses Fraction exterior forms and does not import the first program or read its results. The last verifies all originals, Git state and the new-write scope, then hashes new artifacts. Its deliverable manifest excludes itself. New result files may be regenerated inside this sandbox for reproduction; no prior research is overwritten.

The JSON residual matrices give every entry, not just a scalar pass/fail. All coefficients are exact. No numerical simulation, fitted sample, stochastic ensemble, physical pole calculation, UV completion or new causal theorem is claimed. Explicit smooth-step functions and the connected exact classical solution are derived in report 01; the script verifies the exterior equations for arbitrary smooth f(x),g(x) in the stated closed-coframe construction. The analytic fields A=phi=0 make the remaining curvature/multiplier equations identically zero. The sixteen coframe components do not represent sixteen physical modes, and the rank-two residual span does not represent a particle count.

## Environment and implementation notes

Executed Python: `3.11.13`; SymPy: `1.14.0`. No dependencies installed.

Initial system `python3` lacked `hashlib.file_digest`; that attempt stopped before creating I07. The installed palpt Python was then used for hashing and calculations. During implementation, structural symbolic equality was replaced by simplified polynomial equality; SymPy JSON values were serialized explicitly. Two manually anticipated residual entries used incorrect complementary-pair positions; direct exterior multiplication fixed them to C(01,23)=-1/3 and C(02,13)=-1/6. Both full independent calculations subsequently passed. These corrections did not alter the action, map or negative result.

The connected flat witness was expressed with e0=d[f(x)t] so the two cells can be sampled at equal coordinate time; this sharpens the on-shell witness without changing the parent action or coarse rule. No physical model revision occurred. Preservation is a separate filesystem check, not another physics experiment.

## Exact local source pointers and SHA256

All previous material is read only. The I06 model is a comparison target, not the I07 microscopic action. The original elementary two-coframe sum is acknowledged at I05 report 03, section 4; it is not presented as a newly discovered effect. The new closure proof includes transport, the exact connected solution and the distinction from C19.

- `artifacts/discovery/PTQ_CLOUDS_I05/01_GEOMETRIC_PARENT_AND_VARIATION.md`
  SHA256 `1c2df1988e81405375f586c3355796f13d53a40a4a0c5f477e6221e807a402c3`
- `artifacts/discovery/PTQ_CLOUDS_I05/03_ORIGIN_OF_GEOMETRIC_DYNAMICS.md`
  SHA256 `f480f89ddfed90ba9554c4bf93d6af8376da3686c81793b0e09d0b60fac5d148`
- `artifacts/discovery/PTQ_CLOUDS_I06/01_GEOMETRICITY_AND_RECONSTRUCTION.md`
  SHA256 `8219771a124ef1804dccf96e30d3b5effd478a4bbcae3d5fca3c69a85b3f10f9`
- `artifacts/discovery/PTQ_CLOUDS_I06/02_RELAXED_CONSTRAINT_PHYSICAL_TEST.md`
  SHA256 `e3a2b26c57d7e78d77303b719c712dec732ed2090a6af065dd295f3c799064f9`
- `artifacts/discovery/PTQ_CLOUDS_I06/03_FIRST_PRINCIPLES_ASSUMPTION_VERDICT.md`
  SHA256 `f733d2374fe9b42f24e92561b2d29012d6e03312a2a5fcf50a37b62f56083af4`
- `refs/internal/INTERNAL_WHITEPAPER.md`
  SHA256 `f03c75454717e614c7ff5d8cce0f5ac53ac5f33ac63e6760e4b1d57eb9fb1067`

An external comparison was read through web search only: Dittrich–Martín-Benito–Steinhaus, [Coarse graining flow of spin foam intertwiners, arXiv:1609.02429](https://arxiv.org/abs/1609.02429), abstract. It concerns different spin-net/intertwiner domains. Its qualitative flow result is not used as proof for this real classical construction, and no spin-foam quantization was initiated. No paper was downloaded into historical directories.

The internal predictivity freeze remains effective. No Symmetry audit, IB-2 calculation, empirical fit, manuscript edit, Git mutation, installation, publication, external communication or automatic continuation occurred.

## Preservation records

`initial_state.json` contains the complete baseline SHA256 map of 1235 regular project files (excluding .git), expanded initial untracked-file inventory, HEAD, branch and short status. There were no symlinks. Initially I07 did not exist. Existing short status was `.DS_Store`, `artifacts/`, `refs/` untracked. All new files are confined to I07. `preservation_final.json` is the executed final comparison, not a claim based solely on observing clean Git status; pre-existing untracked evidence is hashed too. `deliverable_sha256.json` hashes final artifacts except itself. No old working tree or PDF archive is copied.
