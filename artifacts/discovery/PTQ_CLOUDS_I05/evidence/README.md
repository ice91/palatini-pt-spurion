# I05 reproduction, source provenance and limits

Run from `/Users/rocky/Downloads/palatini-pt-spurion` with the existing palpt environment (Python 3.11, SymPy 1.14.0). No installation is required:

```sh
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I05/evidence/derive_simplicity.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I05/evidence/derive_modes.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I05/evidence/check_independent.py
/Users/rocky/miniconda3/envs/palpt/bin/python -B artifacts/discovery/PTQ_CLOUDS_I05/evidence/verify_preservation.py
```

All output is written beside the scripts. `-B` avoids bytecode files elsewhere. The final command verifies the 1206 pre-existing regular files, Git HEAD/branch/status and absence of new files outside I05, then hashes deliverables (excluding the hash manifest itself). Rerun it after changing any new artifact.

## What was independently calculated

- **Cycle 1 / `simplicity_results.json`: 21 exact checks.** Lorentzian internal star, all four algebraic branches, signed volume, regular simplicity rank and coframe tangent/kernel equality, degenerate solution/ranks, two separate 24-component torsion/connection maps, a BF shift violating simplicity and the Palatini curvature normalization. Constraint rank at the nonzero rank-one example is six; at zero it is zero; neither is treated as an invertible metric chart.
- **Cycle 2 / `modes_results.json`: 19 exact checks.** Full ten-component Einstein symbol and four gauge vectors, off/null-shell ranks, completeness of gauge plus physical modes, the independent canonical symplectic quotient, linear-Christoffel ΓΓ kinetic/gradient density, kinetic Hessian, double-angle helicities and BF curl/reducibility count. TT substitution occurs after demonstrating the two-dimensional physical quotient.
- **`independent_results.json`: 19 separate exact cases.** This program imports neither SymPy nor the other derivation scripts. Dictionary-form exterior products and Fraction Gaussian elimination check all branches, constraint rank at a non-diagonal coframe, Palatini normalization, nonsimple PT-even BF data, nonclosure under sums, and the full-symbol quotient for an oblique null momentum. This verifies conventions and avoids relying on one axis or one symbolic rank implementation.

The primary derivation scripts and independent program succeeded. No physical model revision was required. There were two substantive derivation cycles, one BF-family model and one pure-BF control. The separately declared scalar example is a classical coupling consistency calculation, not another matter loop or additional BF candidate. Its differential conservation proof and conditional soft calibration are displayed in report 03, not counted as independent experiments.

Algebraic tests do not by themselves prove the nonlinear coframe reconstruction theorem, boundary/global equivalence, positivity of all sectors, full quantum consistency or microscopic emergence. Report 01 supplies the reconstruction argument and its local domain; report 02 supplies the canonical/field-symbol argument. A complete new nonlinear Dirac treatment across rank-changing degenerate sectors was not performed. An unrestricted Lorentz connection is **not** an unrestricted metric-affine connection, and there is no affine-projective gauge introduced here.

## Sources actually consulted, as controls rather than replacement derivations

- `artifacts/discovery/PTQ_CLOUDS_I04/03_FIRST_PRINCIPLES_DECISION.md`: fixed-parent massive-scalar gap and dynamical-metric obstruction. I05 is a newly postulated geometric parent, not a change to I04's scalar spectrum or a rerun of its tests.
- `refs/internal/INTERNAL_WHITEPAPER.md`: constitutive/IB-2 and empirical predictivity remain closed; no original verdict is edited.
- De Pietri & Freidel, [gr-qc/9804071v2](https://arxiv.org/pdf/gr-qc/9804071), §III, Eqs.(5)–(8), Theorem 1 and Eqs.(17)–(23), PDF pp.4–6: original Euclidean sector/classification calibration. Our Lorentzian signs/reality are independently specified and checked; no Euclidean star²=+I is transferred.
- Alexandrov & Krasnov, [0809.4763](https://arxiv.org/pdf/0809.4763), §2, especially pp.5–6: original full Lorentz-group canonical benchmark. Its full nonlinear count is not claimed as a calculation executed here.
- Freidel & Speziale, [1201.4247](https://arxiv.org/pdf/1201.4247), §4, Eqs.(15)–(17): author overview consulted for branch/reality orientation. Original research sources above and the executed derivation provide the technical calibration.
- Weinberg, [Phys.Rev.135,B1049 (1964)](https://doi.org/10.1103/PhysRev.135.B1049): leading soft universality assumes a massless spin-two pole and scattering hypotheses; it is not an existence proof.
- Weinberg & Witten, [original four-page paper](https://hep.physics.uoc.gr/Erasmus-IP-2013/files/lectures/chiral-symmetry/Weinberg-Witten.pdf), Theorem 2, printed p.60: applicability test for a covariant stress tensor carrying physical graviton momentum. No new proof or evasion is claimed.

External papers were read through the web tool; no PDFs or working copies were added to historical refs. Historical Symmetry audits were not repeated. No use of a companion theorem as a proof of this geometric action was made.

## Remaining physical assumptions

The BF variational law, real Lorentz bundle, exact simplicity/trace condition, regular healthy branch, matter metric prescription, κ, zero vacuum cosmological parameter and retarded preparation remain supplied inputs. The field equations, linear mode count, relative kinetic sign and pole then follow. There is no microscopic derivation of simplicity, universal matter content, finite G, a quantum-observer projection, a full nonlinear/quantum stability theorem or a black-hole information mechanism. The ordinary Nieh–Yan identity is kept distinct from Hodge torsion quadratic terms. New research is not retroactive proof of Symmetry.
