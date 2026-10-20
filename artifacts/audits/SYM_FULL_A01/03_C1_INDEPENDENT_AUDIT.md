# C1 independent audit

## Verdict at the original scope

The pure-trace contraction and restricted single-vector no-go survive independent algebra. The published **unique C1 solution of the full stated action, with eta=−alpha3/(4alpha1)>0 under only alpha1≠0, is not established and is contradicted as written**. This is separate from the fact that many nondegenerate algebraic torsion actions can conditionally yield pure trace. No replacement physical action is introduced here.

Sources: PDF pp.4–12 §§2–4, pp.28–33 Apps.A,B; Eq13 includes `M_Pl² R(e,omega)/2`. The implementation is `field_eq.py:38–49`, `connection.py:35–64`, `torsion_decomp.py`, `c1_pure_trace.py`, `pt_even.py`. Independent checks: `tests/test_independent_audit.py`.

## C1.1 — definitions and conventions: CONTRADICTED_AS_WRITTEN

Use the paper's Lorentz signature(−,+,+,+), antisymmetry in the last two torsion indices and trace `t_mu=T^a_{mu a}` (A13). With unit-weight antisymmetrization, Eq16 gives

```
T^a_bc = eta(delta^a_b v_c − delta^a_c v_b),  v=d epsilon
T^a_{mu a} = −3 eta v_mu.
```

This is the opposite sign to Eq10/21, `t=+3 eta v`. The paper itself computes the negative trace for the same seed in A24. Appendix B A14 defines `(g_lmu t_nu−g_lnu t_mu)/3`; its trace is **−t**, so it is not the claimed idempotent projector: on the trace subspace P²=−P rather than P²=P. The correct trace inversion for the stated trace convention has reversed sign. The source `torsion_from_trace` uses that reversed sign and passes pure-trace tests; it therefore differs from printed A14/Eq16 relative to the declared trace lock.

For `S^a=epsilon^{abcd}T_bcd`, with epsilon^0123=+1, epsilon_0123=−1, the printed A15 axial map `−epsilon_lmnr S^r/6` also sends the extracted S to −S. This fails idempotence. Independently the axial norm is `T_axial²=−S²/6` for this S normalization, not `+S²/24` as in A20/A21. A timelike axial test tensor produces a direct exact counterexample. Rescaling q by sqrt2 cannot fix trace or axial errors.

Software has a distinct index issue: `axial_vector` raises only the first epsilon index while contracting with an all-lowered tensor; four inverse metrics are required. The function explicitly comments that the single-index simplification suffices, but it does not. The independent probe uses the source's own epsilon_lower0123=+1 convention and a spatial axial direction. The original pure-trace test has axial zero and cannot detect this defect. `reconstruct_residual` is identically zero by the definition q=T−P−A, even when the purported q is not axial-free; reconstruction alone is not validation of irreducibility. The source epsilon convention is opposite in orientation to the paper's upper-index choice, which is permissible if consistently used, but the missing three index raisings are not a mere orientation convention.

## C1.2 and C1.4 — variation of actual Eq13: CONTRADICTED_AS_WRITTEN / UNDERIVED

Paper Step2 calls Eq20 the torsion-dependent action and keeps only alpha1 I1+alpha3 I3. In metric-compatible Riemann–Cartan geometry, R(e,omega) also depends quadratically on torsion, plus a divergence. It cannot be omitted by treating the connection as independent.

This can be verified without citing a curvature decomposition as an oracle. Adopt

```
D_mu V^a = partial_mu V^a + Gamma^a_{b mu} V^b,
T^a_{mu nu} = Gamma^a_{nu mu} − Gamma^a_{mu nu},
R^a_{b mu nu} = partial_mu Gamma^a_{b nu} − partial_nu Gamma^a_{b mu}
                 + Gamma^a_{c mu}Gamma^c_{b nu} − Gamma^a_{c nu}Gamma^c_{b mu}.
```

For constant flat coframe and pure trace, choose the unique compatible contorsion

```
K^a_{b mu}=(g_bmu t^a−delta^a_mu t_b)/3.
```

Explicit contraction gives `R=−2t²/3`, with all derivative/divergence terms zero. A total divergence cannot remove this constant algebraic term. Exact component calculations also give the axial coefficient `R=S²/24` and q coefficient `R=qtilde²/2`, with q trace-free and axial-free. These are independent evaluations of the original curvature scalar, not new Lagrangian terms.

Consequently in this fully specified standard curvature convention, the trace-dependent part of Eq13 is

```
L_trace = (2 alpha1/3 − M_Pl²/3)t² + alpha3 t.v.
0 = (4 alpha1/3 − 2 M_Pl²/3)t_mu + alpha3 v_mu.
t_mu = −3 alpha3 v_mu / [4(alpha1−M_Pl²/2)]    if denominator≠0.
```

A different overall curvature sign changes which parameter combination appears, but cannot make the missing EH contribution disappear. The printed eta formula follows from varying only Eq20, not full Eq13, unless alpha1 has already absorbed curvature in a specified redefinition. No such redefinition is supplied; Eq13 still explicitly contains R(e,omega). The audit does not silently make one.

The code does no such variation: `T_vec=alpha*d_eps`. `alpha` is externally supplied and never derived from alpha1,alpha2,alpha3 or M_Pl.

## C1.3 and C1.5 — coefficient domain and uniqueness: CONDITIONAL; original sufficiency contradicted

Eq13 declares real alpha1,2,3; alpha1≠0 is only added in §4.1.2. A1–A6 do not restrict all irrep quadratic blocks to be invertible. Using the independent curvature contractions and the stated S definition, the axial block coefficient is `−alpha1/6+M_Pl²/48` and the tensor block coefficient is `alpha1+M_Pl²/4` (qtilde normalization). They can vanish at **nonzero** alpha1. The trace coefficient can vanish at alpha1=M_Pl²/2. These are domain counterexamples inside the nominal arbitrary-real coefficient posture, not proposed parameter changes.

When an unsourced axial block vanishes, axial torsion is not uniquely zero. When the trace block vanishes, a nonzero source may make its equation inconsistent rather than uniquely solvable. Invertibility of the map from delta omega to delta torsion (metric compatibility, fixed tetrad) is a kinematic fact; it does not ensure that the quadratic action's Hessian in those variables is invertible. Lemma3 confuses these distinct requirements if used to infer unconditional elimination.

On the narrower domain where **all actual full-action algebraic blocks are nondegenerate**, no axial/tensor sources occur, and only v_mu sources the trace, the equations can set S=q=0 and align t with v. That is the surviving conditional mechanism. It is not a proof for every alpha1≠0 in the published posture.

## C1.6 — eta positivity: UNDERIVED

The paper explicitly imports eta>0 from the later tensor kinetic positivity claim (§4.1.3 p.12). For a prescribed action, the sign of eta depends on a ratio of coupling constants; the invariant IT depends on eta². The source uses positive test alphas and assigns angle zero. Neither that assignment nor a future C3 positivity assertion independently determines eta>0. At eta=0 the torsion is zero and normalized directional diagnostics also need separate treatment.

## C1.7 — PT versus dynamics: PROVED_WITHIN_DOMAIN for the negative distinction

A scalar PT projector removes odd scalar densities; it does not annihilate an axial tensor just because it is axial. S² is even and survives. The paper correctly states this in §4.1.1. Eliminating S must be a dynamical, nonsingular algebraic equation; the actual equation must include curvature. The code's named-atom parity bookkeeping does not implement a tensor-level projector.

## C1.8 — single-vector no-go: PROVED_WITHIN_DOMAIN

With no tensors besides the Lorentz metric, orientation tensor, and one covector v, an equivariant rank3 tensor antisymmetric in its last two indices and linear in v is a linear combination of

```
g_lmu v_nu−g_lnu v_mu,  epsilon_lmunur v^r.
```

The first has a nonzero trace proportional to v; the second has a nonzero axial contraction proportional to v. Requiring both zero sets both coefficients to zero. Exact contractions independently verify this. Equivalently there is no Lorentz-intertwiner from the vector representation to the 16-component traceless torsion irrep.

This requires linearity and the absence of additional independent background tensors/derivatives. It forbids constructing a *sourced ansatz* for q from a single vector; it does not by itself eliminate homogeneous q solutions of a degenerate field equation. Printed sign errors in A24–A26 do not alter that no-go conclusion because each relevant nonzero contraction still forces its coefficient to vanish.

## C1.9 — scalar projective compensator: CONTRADICTED_AS_WRITTEN

Eq4 allows an **arbitrary one-form** xi. Eq5 requires `delta(d epsilon)=3xi`. But for a local scalar, `delta(d epsilon)=d(delta epsilon)` and thus `d delta(d epsilon)=0`. Therefore xi must be closed locally (and exact globally to admit a single-valued compensator). For example xi=x dy has dxi=dx∧dy≠0, making the stated transformation impossible. Fixed spurion status does not change this integrability identity.

A scalar can compensate exact projective shifts, but the audit does not restrict arbitrary xi to exact xi to save the published symmetry. Global holonomy can obstruct compensation even for closed one-forms. Formal introduction of an independently shifting covector would change the declared fields and is outside this audit.

## C1.10 — metric compatibility versus projective geometry: CONTRADICTED_AS_WRITTEN

With fixed metric and `delta Gamma^a_{b nu}=delta^a_b xi_nu`,

```
delta(nabla_nu g_ab) = −2 xi_nu g_ab.
```

Thus the nonzero affine projective transformation is not tangent to the declared metric-compatible connection space. In tetrad notation its trace shift is not antisymmetric in the two Lorentz indices. It naturally lives in a larger metric-affine space with nonmetricity. This distinction remains even when xi is exact.

Formal projective invariance of a curvature contraction in that larger space, restriction to a subset of projective transformations, and an actual gauge symmetry of Eq13 on the stated variational field space are different assertions. The paper gives no consistent transformation enlarging that space or showing full action invariance. A1–A6 do not resolve this conflict.

## C1.11 — projection and variation: CONDITIONAL, with separate contradicted Appendix identities

For a field-independent real-linear involution U acting equivariantly on the fields **and their variations**, `delta[(1+U)O/2]=(1+U)delta O/2` is valid. Anti-linearity through conjugation commutes with real variation. A3 can consistently be stipulated within an invariant field/background domain. Holding a spurion fixed is not alone sufficient if its PT image is a different, excluded background; the same admissible background orbit or an invariant background must be specified. Metric-dependent Hodge operations also require their genuine variation. This is not an automatic proof of full projected action variation from the string API.

Two additional local failures in Appendix A were independently checked:

1. EqA2 claims `integral Pi[X]Pi[Y]=integral Pi[XY]` for arbitrary real scalars. On a symmetric interval, U:x→−x, X=Y=x: left side0, right side2/3. The true self-adjointness identity is of a different form; the audit does not substitute it into the source. An involution projector is not a multiplicative map.
2. EqA9's Hodge variation uses half the required index-action coefficient for its displayed definition of H.alpha. Under g→exp(2u)g, a fixed 2-form's Hodge star in4D is invariant. Printed A9 predicts a nonzero2u variation. This is an independent formal-calculus error, not a software-environment failure.

## Surviving invariant and operation order

For **imposed consistent pure trace** with magnitude |t|=3|eta v|, exact Lorentzian contractions give

```
T_abc T^abc = 2t²/3 = 6 eta² v²,
IT=−6 eta² Sigma_epsilon.
```

This identity is insensitive to the trace-lock sign error. Its status is PROVED_WITHIN_DOMAIN as algebra, conditional as a consequence of the full dynamics. Reality of the projected scalar is not positivity of its Lorentzian norm.

Eliminating an auxiliary torsion field before metric variation agrees with varying first only when the substituted field is an actual differentiable stationary solution of the **full** action and the relevant block is invertible. The chain-rule term `(delta S/dT)(dT_star/dg)` then vanishes. Substituting a preselected ansatz or Eq21 with omitted curvature does not meet this condition. Neither paper nor code establishes the required commutation over the whole stated parameter domain.

No independent full-action uniqueness proof or universal PT/projective gauge mechanism is recovered. The bounded audit stops at the supplied source; no compensator replacement, coefficient redefinition or new operator is introduced.
