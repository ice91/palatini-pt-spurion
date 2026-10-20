# Reproduce the bounded calculation

Use the already installed interpreter, from the repository root:

```sh
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I01/evidence/derive_observer.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I01/evidence/test_independent.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I01/evidence/verify_preservation.py
```

Python 3, installed SymPy 1.14 and mpmath suffice. No SciPy, production script, project module, network retrieval or new installation is required. `-B` prevents incidental bytecode writes. Results are written only beside these scripts. Read `initial_state.json` as an immutable baseline, not an output to regenerate after execution.

`derive_observer.py` independently differentiates the null-coordinate logarithms, computes pole residue/detailed balance/noise relation, constructs the normalized counterstate, integrates Gaussian kernels by square completion and constructs finite Kraus/Choi/boost matrices. It writes `symbolic_results.json` (26 checks).

`test_independent.py` imports neither that script nor its outputs. It performs 45-digit contour quadratures on two separate nonsingular complex contours, finite switching convolutions, Fock sums, direct y integrations, a covariance calculation, a finite entangled-ancilla positivity test, dephasing/postselection counterexamples and a positive-frequency wavepacket integral. It writes `independent_results.json` (9 unittest methods). Integer/rational exact examples and dimensionless quadrature samples are verification illustrations, not fitted physics. Results prove neither a continuum tensor factorization nor a novel theorem by test count.

The first cycle passed 21 symbolic checks and 8 independent tests. The second cycle added the counterstate and fluctuation-response checks and passed 26/9. No failed derivation/test run is hidden. These counts are not 35 independent physics experiments. The numerical upper-error tolerance for all 10 Fourier checks is 1e-35 absolute; arbitrary precision is 45 digits.

Analytic proofs/caveats are in reports 02–03. The continuum type-III boundary and algebra-wide wedge KMS result are established external input, distinct from computed two-point/finite matrices.

## Primary external references, accessed read-only

- Juárez-Aubry and Louko, *Onset and decay of the 1+1 Hawking-Unruh effect: what the derivative-coupling detector saw*, CQG 31,245007 (2014), §§2,3.2, Eqs.(3.5),(3.8)–(3.11): https://arxiv.org/html/1406.2574v3 . Cross-check of derivative probe and normalization, not a black-hole model imported here.
- Crispino, Higuchi and Matsas, *The Unruh effect and its applications*, RMP 80,787 (2008), §II.9: https://arxiv.org/html/0710.5373v1 . Established algebra-wide vacuum wedge KMS context; direct spectrum derivation is in report02.
- Witten, *Notes on Some Entanglement Properties of Quantum Field Theory*, RMP90,045003 (2018), §§5.3,6.5: https://arxiv.org/html/1803.04993v6 . Established algebra/factorization boundary. No claim to have newly proved type III via a finite regulator.

Only primary author manuscripts were relied on. No external PDF archive was copied. Browser searches included nonprimary results that were not used as mathematical evidence. All local source hashes, including the exact Particles9,93 PDF outside the repo, are recorded in the baseline and rechecked at completion.

## Scope of preservation

The initial manifest inventories pre-existing regular project files excluding `.git` internals, records HEAD, branch, status, Git refs and untracked inventory, and separately hashes the external Particles copy. Final verification checks every pre-existing file, additions outside the sandbox, HEAD/branch/refs/status and tracked/staged diffs. It reports previous discovery/audit/whitepaper preservation without treating them as new experiments. `deliverable_sha256.json` covers final sandbox files except itself; it includes the preservation report. No manifest can include its own hash without a self-reference problem.
