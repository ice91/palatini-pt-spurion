# C1 action evolution, multiplier provenance and conventions

## H03 — the imposed alignment step is recovered, within a changed action [DERIVATION_RECOVERED, narrowly; ASSUMPTION_CHANGED]

Let v_mu=partial_mu epsilon, t_mu=T^nu_{mu nu}. All four versions display the same pre-lock bulk skeleton, in their respective equation numbers:

```
S0 = integral sqrt(-g)[ M_Pl² R(e,omega)/2 + alpha1 I1 + alpha2 I2 + alpha3 I3 ],
I1=Pi_PT[T_ABC T^ABC], I2=Sigma_epsilon, I3=Pi_PT[t_mu v^mu].
```

|Version|Exact original location of S0|Added lock in actual C1 proof|Displayed vector equation|
|---|---|---|---|
|v1|file p.9 Eq11; post-lock Eq12|file p.11 Eq17: sqrt(-g)Lambda^mu(t_mu−3eta v_mu)|file p.12 Eq19:4alpha1 t_mu/3+Lambda_mu=0|
|v2|file p.10 Eq19; post-lock Eq20|file p.12 Eq26: same multiplier|file p.12 Eq28: same vector equation|
|v3|file p.7 Eq8; post-lock Eq9|file p.9 Eq14: same multiplier|file p.9 Eq15: same vector equation|
|Symmetry|p.8 Eq13; post-lock Eq14|**None in main C1 Step2**, p.11 Eq20|p.12 Eq21:4alpha1 t_mu/3+alpha3 v_mu=0|

The narrow recovered derivation is simply

```
delta S / delta Lambda^mu = sqrt(-g)(t_mu−3eta v_mu)=0.
```

Its source is the **augmented** action, its assumption is that Lambda is an independently varied algebraic multiplier, and all steps of this variation are present/explicit. It proves the imposed lock in that constrained theory. It does not show the same lock follows from unaugmented S0, determine eta from its original couplings, or establish the axial/tensor block nondegeneracy. No multiplier is inserted into Symmetry Eq13 by this audit.

For an actual nonsingular unaugmented quadratic trace block with derivative `C t_mu+alpha3 v_mu`, stationarity fixes its own proportionality. An added independent multiplier instead gives `C t_mu+alpha3 v_mu+Lambda_mu=0` and `t_mu=3eta v_mu`; Lambda adjusts to the prescribed eta. These are different variational problems unless an explicit equivalence, including coefficient relation and constraint class, has been proved. No such equivalence is supplied.

## H04 — when the claim changes [ASSUMPTION_CHANGED; PARTIAL_DERIVATION]

- **v1 file p.5 A6** openly states enforced trace lock. File p.12 Theorem2 says “with the trace lock(17)”. But the same page's paragraph “Why the Trace Lock is Not an Extra Assumption” already asserts that the multiplier merely implements a Palatini outcome. Thus the stronger rhetoric already exists in v1; it must not be dated only to v3.
- **v2 file p.12 Proposition4** adds a collinearity argument: one available covector permits only a proportional source/linear ansatz; the current then “simply sets alpha=3eta”. File p.13 Theorem6 still explicitly assumes lock(26). Collinearity is not a proof of the value of its coefficient, nor does it remove homogeneous solutions at a degenerate algebraic block. This is a partial ansatz argument, not a full action derivation.
- **v3 file p.5 A6** now labels alignment a dynamical result and “not an initial constraint”. File p.7 repeats that it is “not a choice”. Yet file pp.9–10 explicitly adds Eq14 and obtains the lock by varying Lambda. Therefore the formal proof remains constrained even after the assumption is relabelled.
- **Symmetry pp.5,11–12** replaces A6 by truncation/fixed spurion and actually removes Lambda from C1 Step2. The alpha3 source reappears and eta is defined by a coupling ratio. Of the available versions, this is the first unaugmented main proof. Section7.2/p.20 still describes p_Lambda and trace-lock constraints, indicating retained material from the earlier augmented action.

There is no evidence of an intermediate complete derivation that made removal of the multiplier innocuous. The change in text is historically trackable; equivalence of the theories is not established.

## H05 — omissions in historical connection variation [PARTIAL_DERIVATION; CONTRADICTED_AS_WRITTEN]

v1 Eq18, v2 Eq27 and v3 Eq15–17 are explicit algebraic irrep equations, so it would be incorrect to say the preprints contain no equations. But:

1. The displayed S0 contains alpha3 I3; their vector variation omits alpha3 v_mu without an explicit Lambda redefinition. A shift Lambda'=Lambda+alpha3 v could change other terms and coefficients; it is not written and is not silently made here.
2. S0 includes R(e,omega). The main irrep variations do not explicitly calculate its torsion-quadratic contribution. Kinematic invertibility of deltaomega→delta torsion is not nonsingularity of the full action Hessian. The old printed alpha1-only equations do not recover the missing full curvature variation diagnosed in A01.
3. v2/v3 AppendixG later acknowledge curvature's trace block, using alpha1 there for a curvature-origin coefficient. No equation maps this use of alpha1 to the free coefficient of I1 in S0. It cannot be imported backwards as an implicit absorption of EH into S0's alpha1 while R(e,omega) remains explicitly present.
4. No version states the complete original axial/vector/tensor singular loci for all real bulk couplings. The restricted single-vector no-go, present in all appendicesB, is not a dynamical elimination proof for an unsourced but degenerate irrep.

The preprints recover an earlier *enforcement mechanism*, not the missing proof of the published unaugmented C1 theorem.

## H06 — exact convention and normalization ledger [VERSION_MISMATCH; CONTRADICTED_AS_WRITTEN]

To avoid overloading the printed T, call the torsion tensor mathsfT, its trace t, and its squared projected norm N=Pi[mathsfT²]. These are only audit labels, not redefinitions of any source action.

|Item|v1|v2|v3|Symmetry|
|---|---|---|---|---|
|Geometry/signature|T^A=De^A; independent compatible antisymmetric spin connection; signature(−,+,+,+)|Same|Same|Same|
|Trace definition|file p.11 Eq17 and p.31 A12: t_mu=T^nu_{mu nu}|file p.12 Eq26 and p.33 A13: same|file p.28 Eq50: same|p.31 A13: same|
|Printed C1 tensor|file p.10 Eq13:2eta delta^A_[B v_C]|file p.11 Eq22: same|file p.8 Eq10: same|p.10 Eq16: same|
|Claimed trace lock|t=+3eta v, imposed by Eq17|t=+3eta v, imposed by Eq26|t=+3eta v, imposed by Eq14|t=+3eta v, claimed from Eq21|
|Main IT definition|file pp.3,10,12:IT=−N/4|file p.3 Eq3, p.10 Eq21:IT=−N/4|file p.7 IT=−I1/4; p.10 same|p.8 Eq12:IT=−I1=−N|
|Printed pure-trace N|file p.10 Eq14; p.33 A28:N=6eta²Sigma|file p.11 Eq23; p.35 A29: same|file p.10; p.30 Eq66: same|p.12 Eq24; p.33 A29: same|
|IT actually implied by that N and own definition|−(3/2)eta²Sigma|Same|Same|−6eta²Sigma in main text|
|IT claimed in reductions|−6eta²Sigma|Same|Same|Same; **AppendixG p.45 A70 still defines IT=−N/4**, contradicting main Eq12|
|eta|Prescribed in multiplier; positivity imported from locked tensor sector|Same; Prop4 only says current fixes coefficient|Same despite new dynamical rhetoric|Main p.12 eta=−alpha3/(4alpha1); AppendixG exact lock would require3eta=1 for the same epsilon|

Consequences:

- The preprint quarter factor and the printed−6eta² relation cannot both follow from the displayed N=6eta²Sigma. This is an internal factor4 error, not a harmless cross-version choice.
- Symmetry's main change from−N/4 to−N fixes that main normalization mismatch, but its AppendixG preserves the older quarter definition. It does not establish action equivalence or fix the missing full connection variation.
- With quarter normalization and the same declared tau=3eta sigma sqrt(|Sigma|), the correct algebraic relation would be `tau²=−6 sigma IT`, not the printed `tau²=−(3/2)sigma IT`. Those are consequences checked within the old conventions, **not substitutions used to repair their claims**.
- All four keep the same trace sign clash: for weighted antisymmetrization the printed tensor `eta(delta^A_B v_C−delta^A_C v_B)` traces as t=−3eta v under their stated t=T^nu_mu nu. Changing version does not remove it. AppendixB's printed trace projector has the corresponding inversion sign issue. Older axial norm/projector problems are not cured by the IT main normalization change.
- In the AppendixG completion (from v2 onwards), the invariant combination is tbar=t−v. Arbitrary-eta main lock gives tbar=(3eta−1)v, not zero. Identifying the two locks therefore requires a stated coefficient/field normalization match; it is not automatic. No such match is supplied.
- v1 has no explicit Stueckelberg AppendixG or arbitrary-one-form projective completion; those are introduced in v2. A01's specific scalar-compensator objection must not be retroactively assigned to a nonexistent v1 completion. From v2 onwards the written arbitrary-one-form law and compatible-connection conflict persist.

The common sign placement in the rank-one matrix is audited in04. It is essential for comparing literal actions, especially timelike spurion gradients.

## Effect on SYM_FULL_A01

**Provenance is materially clarified:** the journal's unexplained multiplier references in the DoF section have an identifiable earlier action source, and v3's dynamical wording is inconsistent with its retained multiplier proof. **The central missing proof is not recovered:** these earlier extra constraints are not part of the journal main Eq13 and cannot be imported as rescue assumptions. All changes are recorded in this new audit; the previous audit remains immutable.
