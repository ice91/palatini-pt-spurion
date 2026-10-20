# D02 — physics, novelty, historical boundaries and final decision

**NEW MODEL decision:** the fixed geometric parent has an action-derived nonzero torsion response and two justified PT-compatible retained maps. They do **not** generate two inequivalent physical sectors under the fixed-parent protocol. The exact retained/complement descriptions reconstruct the same physical responses. No PT-specific restriction beyond the controls survives.

## 1. What was actually discovered/calculated?

The one parent action (1) gives

    t_μ=(3β/(2ℳ²))∂μB,
    T^a_{μν}=(δ^a_νt_μ−δ^a_μt_ν)/3,
    K=I−(3β²/(2ℳ²))uu^T,
    3β²<2ℳ² for two positive kinetic scalar directions.

These are **NEW CONDITIONAL MATHEMATICAL RESULTS** of the stated new action. Axial/tensor torsion were not deleted or assumed zero; the 24-variable connection Hessian is nonsingular and the full stationary solve gives the displayed branch. There is no propagating auxiliary torsion mode. The resulting geometric relation t∝dB is genuine content of this model, not an arbitrary plotting prescription.

P1 retains a species line; P2 retains the line singled out by the torsion current. Both are K-self-adjoint idempotents compatible with the real/PT structure. At 0<κ<1,a≠b both have nonzero complementary mixing. Their exact conditional Schur operators retain two physical poles, sources and memory, whereas strict deletion loses information. Matrix identity (23) proves equality for all matched parent source/detector vectors; it is not an equality inferred from similar numerical plots.

This improves on D01 in an important bounded respect: geometry and its connection are actually dynamical in the parent and an explicit matter current sources the connection. We still use a flat zero-field solution for the tractable linear response comparison. There is no claim of a solved nonlinear torsion-rich cosmology or a complete observable theory of gravity.

## 2. Controls that remove apparent PT novelty

| Control | Explicit calculation / proof | Result |
|---|---|---|
| A. Ordinary geometric covariance | U=θ*conjugation transports real e,ω,Φ, d, wedge, metric and volume; t·dB is a natural real scalar | Every constant β,a,b has the same covariance. PT is not the source of the trace coefficient or positivity bound |
| B. Generic real-linear/anti-unitary Z2 | Both Pi are real constant endomorphisms and commute with the same conjugation/pullback; idempotence is line projection in K | The same construction works without interpreting the involution as physical spacetime PT. No exclusive PT coefficient equation appears |
| C. Ordinary positive Hermitian control | For κ>0, put ψ=K^(1/2)Φ. The exact reduced action is Einstein gravity plus canonical real scalars with symmetric positive mass matrix H=K^(−1/2)diag(a,b)K^(−1/2); diagonalize H orthogonally | A regular constant local field redefinition reproduces all scalar/metric responses and characteristic cones with transported sources. Both projector charts are ordinary canonical coordinates |
| D. Ordinary auxiliary elimination | Shift C to C−C*(e,dB); the connection action completes its quadratic square and leaves (6), plus the explicit boundary | Sourced torsion and the kinetic correction are standard action-derived auxiliary response, not a new symmetry-generated law |

The connection shift is triangular and local, with C* depending on scalar first derivatives; it is invertible when e is nondegenerate and ℳ²≠0. It preserves the compatible connection type because both C and C* have antisymmetric lowered Lorentz indices. The scalar canonicalization is derivative-free and regular for κ>0. Sources and original observables must be transformed, not discarded.

In particular, the original geometric trace detector in the control is the reconstructed composite t=(3β/(2ℳ²))dB. A control that simply sets that same detector to zero would not preserve the observable dictionary and is not the equivalence claimed. The theorem concerns the declared classical parent/source algebra; no quantum measure/Jacobian or interacting quantum-gravity equivalence is asserted.

## 3. Adversarial falsifications and evidence

| Strong claim tested | Counterexample / derivation | Scientific outcome |
|---|---|---|
| Two PT-compatible idempotents create two physical worlds | The same complete response matrix is Ti Di^(−1)Ti^T=D^(−1) for both | Refuted within this fixed-parent, matched-data comparison |
| A retained line is automatically autonomous | μ1=−ah/(c√κ), μ2=(a−b)/(2√κ) are both nonzero generically | Both candidates are effective charts, not closed sectors |
| Different intrinsic responses imply inequivalent sectors | Replacing v=Kn1 by Kn2 changes the source and detector; the poles stay fixed while weights change | DIFFERENT_PREPARATIONS_ONLY |
| Strict projection equals exact reduction | Ei=Ai−μi²/Fi; correction and P2 source contact are nonzero | Strict deletion is a distinct approximation, not a theorem about the parent |
| A Schur chart singularity is a new physical mode | Eq.(25) gives nonzero full determinants at Fi=0 | Chart problem only; complete responses/other charts must be used |
| A zero kinetic eigenvalue is an extra gauge mode | κ=0 gives −sB−dD=J_B and one remaining wave with mass 2ab/(a+b) | Actual algebraic constraint, not an arbitrary gauge function |
| PT guarantees positive physics | κ<0 retains natural PT covariance but has one negative kinetic direction | Positivity is a computed consistency requirement, not a PT consequence |
| A retarded effective theory is fixed by anti-linear PT | Retarded oscillator kernel maps to advanced | Action covariance does not imply same-state inverse invariance |
| A trace branch proves original C1 and three-chain equality | New coefficients follow from this full curvature action; no original route densities were changed | No retroactive proof or repair |

**Executed evidence:** `geometry_results.json`: 9 exact checks, rank(H0)=24, det(H0)=256; `projection_results.json`: 19 exact checks including full source/detector matrix equality; `independent_test_results.json`: 11 tests, all pass. The independent script imports neither derivation nor saved results. It verifies all 24 stationary connection variations, derives scalar EL equations from a separate jet action, checks the NY/norm distinction, solves the same physical sources in both charts, and tests contact omission, closure, spectral weights and the critical constraint. One source-list formatter fix and one polynomial-factor normalization fix are retained in the execution notes/attempt counts. Neither was a model revision. Test success supplements the displayed analytic proofs; it is not alone a physics gate.

## 4. Relation to C1 and C2 only AFTER geometry was derived

For our exact stationary torsion,

    𝕋²:=T_ABC T^ABC=(2/3)t²
       =(3β²/(2ℳ^4))(dB)².                               (27)

Therefore the induced density (3β²/(4ℳ²))(dB)² lies on a genuine single scalar derivative direction. This is a derived sourced trace branch of the NEW action, with β and the EH coefficient fixing its alignment; it is not Symmetry's ε spurion or its η coefficient. Here B is varied matter, unlike the fixed background ε of the historical paper.

There is a limited common operator direction with the post-pure-trace ROD/CM quadratic expressions: on a real sign-fixed non-null gradient patch, their torsion amplitude squared reduces to a coefficient multiplying |dB|², hence to the same line as (dB)² after the patch sign is fixed. This is **E1 only**. No coefficients or complete actions are identified, no determinant normalization is adjusted, and no three-route theorem is reconstructed. At the actual zero-field background dB=0 the normalized n=dB/√|dB²| used by those constructions is not itself defined; our polynomial induced density is smooth there. That distinction forbids using the comparison as an action-level dictionary at the vacuum.

Ordinary Nieh–Yan is

    NY=d(e_A∧T^A)=T_A∧T^A−e_A∧e_B∧R^{AB}.

For pure trace T^A=(t∧e^A)/3, e_A∧T^A=0 and T_A∧T^A=0, so NY=0. This does **not** make the Hodge norm zero:

    T^A∧*T_A=(1/2)𝕋² vol_g=(1/3)t² vol_g,

which is generically nonzero. The independent wedge/component test confirms this distinction. No absent CS+ bulk density is invented or added to the parent. A nonzero norm does not arise by applying a final star to zero NY.

Historical reference boundaries: Symmetry `refs/symmetry-18-00170-v2.pdf` p.8 Eqs.(9)–(14), pp.10–12 C1 and p.14 Eqs.(26)–(28); completed `artifacts/audits/SYM_FULL_A01/03_C1_INDEPENDENT_AUDIT.md`, `04_C2_THREE_ROUTE_AUDIT.md`; the literal timelike qualification remains in `artifacts/audits/SYM_HIST_P01/04_C2_C3_SOURCE_RECOVERY.md`. Their findings are not redone, overwritten or treated as new experiments. T01's distinction between averaging, carrier maps and inverse-prescribed Schur is preserved.

## 5. IB-2 and JGP firewall

**NO_ACTION_LEVEL_DICTIONARY** has been established between this new two-dynamical-scalar/compatible-connection action and the frozen constitutive pair (f1(φ),Z_geom(φ)). Our constants and κ are not those two arbitrary functions. The relation κ=1−3β²/(2ℳ²) is a derived constant kinetic correction inside a chosen finite model, not an independently sourced functional equation E[f1,Z]=0 on the canonical function space.

Choosing constant functions by hand would specify a small toy subfamily, not reduce the allowed canonical constitutive family by a derived law. No positive functional constraint rank on IB-2 is demonstrated; in the absence of a valid dictionary a rank-transfer claim is not applicable. Promoting our varied B to the whitepaper's prescribed scalar, or transferring its equation into that frozen theory, would change the problem rather than close it.

`refs/manuscript.pdf` pp.3–4 Eq.(3.7) and p.13 Theorem 7.2 use a distinct metric-affine trace/vector domain and varied one-form compensator. That action is neither substituted for this compatible geometry nor used as a revision of Symmetry. Its scalar Keff is not our gravitational tensor coefficient. Internal whitepaper §§1,7,9,13,15–19 is unchanged: **the constitutive-predictivity DEEP_SCIENCE=CLOSED freeze remains effective**, including IB-2, SPARC/cosmology fitting and H_ref/C_PTQ calibration. This separately authorized foundational sprint is not a reopen trigger.

## 6. Final scientific classification

- **Derived new geometry restriction:** yes, within the new fixed action: all compatible auxiliary components are solved; torsion is sourced pure trace, t=(3β/(2ℳ²))dB, with the positive scalar kinetic condition 3β²<2ℳ². Generic geometry/auxiliary dynamics account for it; PT does not select it.
- **Two meaningful PT-compatible retained maps:** yes. **Two distinct autonomous admissible physical sectors:** not established for those maps; both fail closure generically.
- **Matched-data exact classification:** SAME_REPRESENTATION_CLASS. Different intrinsic line preparations: DIFFERENT_PREPARATIONS_ONLY. Strict truncation claimed to be exact or autonomous: NOT_WELL_DEFINED as that claim.
- **Schur information:** nonzero and important; it retains complementary propagation/source/history that strict deletion loses. It supplies no new parent spectrum merely by changing projectors.
- **PT specificity:** no nontrivial physical restriction survives covariance, generic real/anti-unitary structure, Hermitian control and ordinary auxiliary elimination.
- **Functional closure:** no positive IB-2 rank result and no canonical theory reopening.

This is a completed limited no-go plus a sourced geometric example, not merely a missing-formula audit. The new parent is consistent on a nonempty regular domain; the attempted projection-induced physical multiplicity is what fails the matched-observable test. There is no claim that all possible theories lack genuinely distinct physical/superselection sectors.

## 7. ONE next mathematical computation — not executed

The most informative next calculation would be the **constraint-reduced physical principal symbol about one explicitly specified nonzero-gradient solution of this SAME parent**, including metric/scalar mixing, with both projections evaluated on that same solution. That tests the current flat-response limitation on an actually torsion-sourced background. It should be compared with the exact Einstein–scalar control; there is no prior reason to expect a PT-specific effect. This is one proposed future computation, not a new task, automatic continuation or an IB-2 reopening.

## 8. Preservation and stop

Initial HEAD `6b456b89baa866b6dd05bc69a837b98a985a34ad`, branch `main`; full initial SHA256/state inventory in `evidence/initial_state.json`. The target directory was absent and created once. Only these three reports and new evidence/scripts were written in D02. Final hashes/state and deliverable integrity are recorded in `evidence/preservation_final.json` and `evidence/deliverable_sha256.json`. No original code, config, tests, references, manuscript, prior audit/gate/discovery result or whitepaper was changed. No Git mutation, dependency installation, network action, external communication, fitting or publication occurred.

**NEW RESEARCH != RETROACTIVE PROOF OF SYMMETRY. STOP after derivation, adversarial review and preservation verification. No automatic continuation.**
