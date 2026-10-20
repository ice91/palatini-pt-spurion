# Independent exact checks

Final execution: **40 passed, 0 failed**, including four successful counterexample checks. A successful counterexample check is an audit success and can establish a manuscript claim failure. The two Appendix J counterexample checks address one underlying issue; the boundary check demonstrates the scope of a stronger statement the manuscript does not claim.

No original project test or module was used as an oracle. No source Hessian, target reduced-action value, or proposed stationary solution was fed into the derivation. The source action, tensor conventions, dimension, signature, and trace-only domain are legitimate physical/mathematical inputs. Target formulas appear only in comparisons after those quantities have been derived.

The symbolic calculation retains arbitrary f1, f2, p=df1 and k=df3. Since the fi are functions of one scalar, physical p and k are collinear where applicable; none of the detected errors depends on treating them as noncollinear. For example the G.5 residual is already nonzero for one shared spatial gradient. Singular-branch compatibility is computed before inversion. The Lorentz-pairing identity is checked on the entire 16×16 Hessian, so a one-component species solve is not mistaken for a four-component test.

The full finite projective curvature test keeps an arbitrary 64-component background connection and arbitrary first jets of xi. Its 256 component identities are polynomial checks, including cancellation of mixed and quadratic terms; these are not 256 independent physical experiments. A is an independent one-form.

## Final checks

| # | Check | Result | Category / evidence |
|---|---|---|---|
| 1 | trace-coordinate map invertible | PASS | IDENTITY — 108 |
| 2 | distortion reconstructs all three original trace conventions | PASS | IDENTITY |
| 3 | curvature current from independent contractions | PASS | IDENTITY |
| 4 | curvature quadratic comparison with Eq5.1 | PASS | IDENTITY |
| 5 | curvature derivative equals divergence of reconstructed W | PASS | IDENTITY |
| 6 | finite arbitrary-xi curvature law all 256 components | PASS | IDENTITY |
| 7 | arbitrary-xi scalar curvature invariant | PASS | IDENTITY |
| 8 | full affine bulk exactly invariant for arbitrary one-form | PASS | IDENTITY |
| 9 | nonexact xi is permitted without scalar integrability | PASS | IDENTITY — Independent antisymmetric xi jets retained throughout curvature test; A is an independent one-form. |
| 10 | quotient density contains no V | PASS | IDENTITY |
| 11 | full one-form Hessian includes varied A and Lorentz pairing | PASS | IDENTITY |
| 12 | horizontal full Hessian has Lorentz pairing | PASS | IDENTITY |
| 13 | raw gauge is exact Hessian null vector | PASS | IDENTITY |
| 14 | raw compatible source / off-shell Noether identity | PASS | IDENTITY |
| 15 | horizontal determinant comparison Eq6.1 | PASS | IDENTITY |
| 16 | regular raw species rank3 exact kernel1 | PASS | IDENTITY |
| 17 | regular full fibre rank12 kernel4 | PASS | IDENTITY |
| 18 | Palatini block determinant comparison Eq5.16 | PASS | IDENTITY |
| 19 | Schur coefficient derived as 2f2 | PASS | IDENTITY |
| 20 | Schur source derived as -df3 | PASS | IDENTITY |
| 21 | stationarity includes all four varied raw species | PASS | IDENTITY |
| 22 | raw and quotient on-shell values agree | PASS | IDENTITY |
| 23 | derived reduced scalar coefficient comparison Eq7.1 | PASS | IDENTITY |
| 24 | S2 ranks and compatibility computed without division | PASS | IDENTITY — compatibility: [p, -p]; full rank 4; full nullity 12 |
| 25 | S2 compatibility iff df1=0 | PASS | IDENTITY |
| 26 | S2 q,u free and t=df3/(2f2) | PASS | IDENTITY |
| 27 | S3 ranks and compatibility computed without division | PASS | IDENTITY — compatibility: [-k]; full rank 8; full nullity 8 |
| 28 | S3 compatibility iff df3=0 | PASS | IDENTITY |
| 29 | S4 ranks and compatibility computed without division | PASS | IDENTITY — compatibility: [p, -p, -k + 2*p]; full rank 0; full nullity 16 |
| 30 | S4 compatible iff both df1=df3=0 | PASS | IDENTITY |
| 31 | raw to horizontal congruence | PASS | IDENTITY |
| 32 | horizontal coordinate determinant transforms by square | PASS | IDENTITY |
| 33 | on-shell current Wstar=-3df1/f1 | PASS | IDENTITY |
| 34 | counterexample: literal EqG5 has extra df1.df3/f2 | PASS | COUNTEREXAMPLE — k*p/f2 |
| 35 | AppendixJ Schur formulas correct | PASS | IDENTITY |
| 36 | AppendixJ source formula correct | PASS | IDENTITY |
| 37 | counterexample: pointwise uniqueness in AppendixJ fails for df1=0 | PASS | COUNTEREXAMPLE — Constant nonzero f1, alpha=4f1, beta=0; both fingerprint changes vanish. Universal-jet interpretation avoids this counterexample. |
| 38 | AppendixJ counterexamples also exist arbitrarily near origin | PASS | COUNTEREXAMPLE — At df1=0 take alpha=4f1(4+m)/(4-16m-11m^2), beta=m alpha, m approaching -4 but unequal to -4. |
| 39 | AppendixJ uniqueness as identity for arbitrary source jets holds | PASS | IDENTITY |
| 40 | counterexample: projective invariance does not remove variational boundary flux | PASS | COUNTEREXAMPLE — At a flat non-null boundary with df3=0, delta t_normal=1 gives delta(J3.normal)=f3. Bulk theorem must retain its boundary hypothesis. |

## Execution record

| Command / stage | Exit | Outcome |
|---|---|---|
| `PYTHONDONTWRITEBYTECODE=1 python3 .../evidence/derive_from_action.py` | 0 | Curvature, full Hessian, stationary solutions and reduced scalar derived. Stdout/stderr preserved. |
| First validation attempt | 1 | Audit harness TypeError: projective function shadowed the Schur scalar name. Preserved as `validation_attempt1.*`; original sources unaffected. |
| Corrected validation, before adding nearby-conic check | 0 | 39/39; preserved as `validation_attempt2.*`. |
| Final validation | 0 | 40/40; `validation.stdout.txt`, empty `validation.stderr.txt`, structured `independent_validation.json`. |
| `PYTHONDONTWRITEBYTECODE=1 python3 .../evidence/verify_preservation.py` | 0 | All 1029 baseline files unchanged; no unexpected file outside this audit; HEAD and branch unchanged. |

The only revised code was the newly authored audit harness. No dependencies were installed. No numerical values were manufactured for missing physical coefficients. Stability, physical DoF, ADM/GW dispersion, and unrestricted MAG claims were not artificially tested: the source does not supply or claim their derivations. These absent tasks are explicitly adjudicated as MISSING_SOURCE for any such extension, rather than being counted as failed auxiliary tests.

Reproduction scripts and full machine-readable results are in `evidence/`. Main derivation and claim-level consequences are in `01_INDEPENDENT_DERIVATION.md` and `02_CLAIM_ADJUDICATION.md`.
