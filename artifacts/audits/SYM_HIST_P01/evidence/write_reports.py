from pathlib import Path
import json
O=Path(__file__).resolve().parents[1];E=O/'evidence';meta=json.loads((E/'pdf_metadata.json').read_text())
def write(name,t):(O/name).write_text(t.strip()+'\n')
rows=[]
for v,date,title in [('v1','2025-09-29','Palatini–PT-Even Torsion: Uniqueness, Three-Chain Equivalence, and Exact Luminal Tensors'),('v2','2025-10-06','A Palatini PT-Even Torsion Framework: Conditional Uniqueness, Bulk Rank-One Equivalences, and Tensor Luminality via Coefficient Locking'),('v3','2025-10-30','Guaranteed Tensor Luminality from Symmetry: A PT-Even Palatini Torsion Framework'),('symmetry','2026-01-16','Guaranteed Tensor Luminality from Symmetry: A PT-Even Palatini Torsion Framework')]:
 m=meta[v];rows.append(f'|{v}|{date}|{m["pages"]}|`{m["file"]}`|{title}|')
write('01_VERSION_CHRONOLOGY.md',r'''
# SYM_HIST_P01 — version chronology and source boundary

## Decision in brief

**C. Historical files confirm unresolved gaps.** The four available PDFs and accessible Git history recover the earlier *constraint-enforced* action and several missing provenance facts, but no complete historical action-to-ADM-to-K,G-to-constraints derivation. The chronology materially refines SYM_FULL_A01; it does not reverse its central negative/conditional verdict. Earlier constraints are not imported into the final paper as repairs.

## Authoritative files actually present

All four requested files were found directly in `/Users/rocky/Downloads/palatini-pt-spurion/`; no guessed path or replacement repository was used. All PDF-page references in these reports are **physical file pages, including the preprint cover**. Thus v2 file p.46 is printed45 of48; v3 file p.41 is printed40 of43. Symmetry file and printed page numbers coincide. v1 has a cover and44 total pages, with no consistent body pagination visible on the cited early pages.

|Version|Posted/published date on source|PDF pages|Exact local filename|Title|
|---|---|---:|---|---|
'''+ '\n'.join(rows)+r'''

Dates above are publication/header dates, not inferred from Git commit times. The three Preprints PDFs have embedded creation dates on2025-11-11 (v1 01:46:14Z; v2 04:25:40Z; v3 08:36:05Z). These later file-processing timestamps do not prove that the mathematical content was changed on that date. Local filenames, DOI/header identities and metadata were checked; no claim of byte identity to the bytes originally served on each September/October posting date is made. The supplied local PDFs themselves are the authoritative evidence for this task.

SHA-256 values:

'''+ '\n'.join(f'- `{m["file"]}`: `{m["sha256"]}` ({m["bytes"]} bytes).' for m in meta.values())+r'''

Metadata and text: `evidence/pdf_metadata.json`, `evidence/{v1,v2,v3,symmetry}/page_XX.txt` and `full.txt`. Seventeen selected pages were also rendered and visually checked for fraction factors, sign placement and equation labels; rendering commands are retained. The Symmetry PDF hash matches the previously audited publisher-identical PDF in SYM_FULL_A01.

## H01 — action and assumption chronology [ASSUMPTION_CHANGED]

|Stage|Actual source content|Effect|
|---|---|---|
|v1, file pp.5,9,11–13|A6 explicitly enforces trace lock; bulk Eq11 augmented by multiplier Eq17; vector Eq19 includes Lambda. Theorem2 explicitly says “with the trace lock”. Prose nevertheless already says it is not an extra assumption.|Constrained action is the mathematical input, despite stronger explanatory rhetoric.|
|v2, file pp.5,10,12–14|Same A6; bulk Eq19 plus multiplier Eq26; new Proposition4 argues collinearity; Theorem6 retains lock; §6.2/file p.18 explicitly imposes K=G as conditionL2.|No removal of the constraint and no independent physical K,G derivation.|
|v2, file pp.45–47|AppendixG first appears: new Stueckelberg action A61, finite algebraic equation A62, asserted exact solution A63, separate hard limit/current A64.|Additional action and limiting prescription are actually supplied, but the finite-mass exact-solution inference is false.|
|v3, file pp.5,7–10|A6 now calls trace alignment a dynamical result, “not an initial constraint”; bulk Eq8 still augmented by multiplier Eq14; vector Eq15 and deltaLambda constraint remain.|Narrative/assumption reclassification precedes actual removal of multiplier.|
|v3, file pp.40–42|AppendixG essentially repeats v2, renumbered97–103; the same finite-mass problem remains.|No recovered finite-mass proof.|
|Symmetry, pp.5,8,11–12|A6 becomes two-derivative truncation/fixed spurion. Main C1 proof drops multiplier and varies alpha1 I1+alpha3 I3, Eq20→21, defining eta=−alpha3/(4alpha1).|First of these four sources with an explicitly unaugmented main C1 proof and a coupling-derived eta formula. The full EH contribution is still not included there.|
|Symmetry, pp.20,44–45|Hamiltonian narrative still refers to trace-lock multiplier; AppendixG still has a separate exact constraint/hard-limit implementation.|Residual mismatch between main proof, DoF story and supplementary completion.|

This dates the transition only within the available versions. No inaccessible intermediate manuscript is inferred. See02 for the complete action/variation comparison.

## H02 — accessible Git chronology [CONFIGURED_DEMONSTRATION; VERSION_MISMATCH]

Read-only Git commands recover **72 reachable commits**, **130 historical paths**, **256 unique file blobs**, of which **244 decode as UTF-8**. `git fsck --full --no-reflogs --unreachable` exited0 with no unreachable objects reported. The clone is not shallow. Local refs are main/origin refs; no version-matching preprint tags or separate LaTeX branch were found. No fetch, checkout, reset, object write, ref update or remote mutation was used.

|Author date / commit|Evidence recovered|
|---|---|
|2025-09-15 `c10b3310cad3d5354908f2cc3b636e3394b99305`|Earliest field_eq implementation directly sets T_vec=alpha*d_eps; no connection variation.|
|2025-09-15 `662fda4ec60c3c220c25732f9b5bf4661599315b`|Early quadratic_action defines a chosen linear speed combination and positive normalization.|
|2025-09-18 `90495ef560638471c384a56e5999fb17cca8728b`|NLO parameterization introduced; no historical change to that module's sole blob.|
|2025-09-21 `1dfaf83279c378e1f8321661113d48e26af0c475`|Mixing config introduced as a grid/fit recipe with comments and symbolic:false; extraction script is an empty blob.|
|2025-09-21 `b24e203034c419b7a2cde2f8a8c4c823be58b669`|First nonempty mixing script stages JSON; says a compute path can be added later.|
|2025-09-21 `925736072ac3f9e89ca50c8dafa6b9c029ddb008`|Config replaced by explicit four numbers, including CM(divN)=+.35.|
|2025-09-21 `55f14b84668d9397ac0efb3578b3c4676e6ce96a`|CM(divN) edited to−.13333333333333333; ratio-checking/meta logic added in same commit. No ADM calculation supplied.|
|2025-09-22 `84501317edf5003e72cf54818075e4319bf9dcf9`|Last available2025 commit; earlier than all three posted preprint dates.|
|2026-01-23 to2026-01-29, ending `6b456b89baa866b6dd05bc69a837b98a985a34ad`|Citation/docs updates; no recovered original action-variation pipeline.|

Commit timestamps are recorded provenance, not independently certified execution dates. The same September code cannot be declared a convention-matched artifact for every later PDF without a version map. Relevant source files' historical alternatives were inspected, including all7 quadratic_action blobs,3 field_eq blobs,5 coeff_extractor blobs,4 degeneracy blobs and all mixing-script/config blobs.

Only two historical LaTeX-related assets exist: `tex/tikz/figA_basis_flow.tikz` and `tex/tikz/figB_conn_decomp.tikz`, both diagrams. **No manuscript .tex, .sty/.cls or mathematical derivation LaTeX archive is available in any recovered tree.** No claim is made that such a private archive never existed.

Full command stdout/stderr/exit codes: `evidence/git_commands.json`. Complete tree/blob inventories, read-only blob snapshots, chronological source patches and notebook cells are in evidence. No old code or notebook was executed. Nine unique historical notebook blobs contain no saved outputs and null execution counts; this does not prove they were never executed elsewhere, but no execution evidence is recoverable here.

## Preservation controls

Initial cwd/HEAD/branch/runtime and SHA-256 of **every pre-existing regular file outside .git**, including the entire existing SYM_FULL_A01 audit, were recorded before extraction. Existing files and prior audits were not edited. New reports/evidence live only in `artifacts/audits/SYM_HIST_P01/`. No dependencies were installed; the limited equation checks use existing Python3.9.6/SymPy1.14.0 and do not execute project code. Final preservation verification is in `evidence/preservation_final.json` and06_RESEARCH_DECISION.md. Git internals are observed read-only through Git commands; scientific-file hashes do not purport to hash live .git metadata.
''')
write('02_C1_ACTION_EVOLUTION.md',r'''
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
''')
write('03_STUECKELBERG_APPENDIX_AUDIT.md',r'''
# Stueckelberg AppendixG: finite mass versus exact constraint

## H07 — exact sources and present/missing steps [PARTIAL_DERIVATION]

|Object|v2 source|v3 corresponding source|Symmetry corresponding source|
|---|---|---|---|
|tbar=t−d epsilon|file p.45 A60|file p.40 Eq97|p.44 A65|
|S_Stk|file p.45 A61|file p.40 Eq98|p.44 A66|
|Printed trace equation|file p.45 A62|file p.41 Eq99|p.44 A67|
|Asserted exact t=d epsilon|file p.46 A63|file p.41 Eq100|p.44 A68|
|Hard penalty/current alternatives|file p.46 A64|file p.41 Eq101|p.44 A69|
|Low-energy action/quarter IT|file p.46 A65|file p.41 Eq102|p.45 A70|
|Weighted epsilon NY addition|file p.46 A66|file p.41 Eq103|p.45 A71|

v1 ends at AppendixF and has no corresponding AppendixG. Equation numbering is not interchangeable: v3 uses plain97–103, and Symmetry A61–A65 belongs to other material before its AppendixG. This report follows the numbers actually printed in each file.

All three later versions provide an explicit action

```
S_Stk = integral sqrt(-g)[M_P² R(e,omega)/2
          −(m_T²/2)(t_mu−v_mu)(t^mu−v^mu)
          −(f_epsilon²/2)v_mu v^mu],  v=d epsilon.
```

They state m_T>0, f_epsilon≥0, A1–A5, discard admissible divergences, and say alpha1 in the following equation comes from the curvature trace-quadratic piece. The existence of this written action and trace equation is recovered source material. What is absent is an evaluated curvature coefficient with unambiguous match to the main action's independent alpha1, and a correct solution step.

## H08 — the written finite-mass solution is false [CONTRADICTED_AS_WRITTEN]

Grant the paper its own printed trace equation, without assuming that its coefficient/sign has already been derived from S_Stk:

```
m_T²(t_mu−v_mu)+(4alpha1/3)t_mu=0.
```

Writing c=4alpha1/3 only as an algebra abbreviation, exact solution for finite m_T is

```
t_mu = [m_T²/(m_T²+c)]v_mu,         m_T²+c != 0,
tbar_mu = −[c/(m_T²+c)]v_mu.
```

Substitution of the asserted t=v into the written equation instead leaves residual `c v_mu`. Therefore the source's exact solution holds only on exceptional branches such as c=0 or v=0, or as the stated hard-penalty limit under appropriate coefficient behavior. No such branch restriction accompanies A63/Eq100/A68. Positive m_T² by itself also does not imply uniqueness: if c=−m_T², the coefficient vanishes. For nonzero v the trace equation is then inconsistent; for v=0 it loses trace uniqueness. This is an algebraic diagnostic, not a choice of new physical coefficients.

With finite fixed c and v,

```
limit_(m_T² -> infinity) t_mu = v_mu,
tbar_mu = O(c/m_T²).
```

This explains which **limiting** enforcement the old action can support; it does not turn the finite-mass solution into the asserted exact equality. E≪m_T alone is not the same limit as m_T²≫|c|, nor does it set the residual exactly to zero. With the same epsilon as the main lock, its finite proportionality would have to match3eta=m_T²/(m_T²+c); no version derives this relation as the original main-action eta.

Every algebraic step above is checked with existing SymPy in `evidence/verify_written_algebra.py` and `algebra_checks.json`. These are scoped checks of printed equations, not a recovered physical theorem.

## H09 — actual added action and limiting domain [ASSUMPTION_CHANGED]

Expanding the given mass term yields

```
−(m_T²/2)t² + m_T² t.v −[(m_T²+f_epsilon²)/2]v².
```

It therefore specifies an additional trace-sector completion and related coefficients; it is not a proof for arbitrary independent alpha1,2,3 in the main S0 skeleton. Main S0 uses the full tensor I1, not merely the trace t², so equality of complete axial/tensor sectors also needs a map. None is supplied. The audit neither adjusts these coefficients nor inserts this completion into journal Eq13.

The alternative `L_Lambda=sqrt(-g)Lambda^mu(t_mu−v_mu)` gives tbar=0 exactly by deltaLambda. This is a legitimate **constraint-enforced step** under a different variational problem. It agrees with the hard limit's stationary constraint, not with generic finite-mass stationarity. It corresponds to3eta=1 only if epsilon/trace conventions are identical; no arbitrary-eta equivalence follows.

At finite mass, integrating an algebraic quadratic variable is not an exact functional delta imposing tbar=0. An exact delta requires an actual multiplier integration or a carefully defined constraint limit; in Lorentzian signature, a “penalty” term is oscillatory rather than automatically a positive Gaussian. The source supplies no path-integral measure/normalization/contour calculation establishing its stronger delta-function statement. The classical limiting stationary equation is the narrower conclusion used here.

A61/Eq98 has no displayed torsion derivative kinetic term beyond the curvature divergence/algebraic split, so the quoted O(partial^4/m_T²) effective action estimate is not an explicit calculation of such operators. The algebraic elimination can leave two-derivative v² terms; taking f_epsilon²→0 is not by itself proof of a nondynamical scalar or zero induced kinetic term. Holding epsilon fixed is a separate declared choice, which is not changed by this audit.

## H10 — symmetry and boundary completion does not close old gaps [CONTRADICTED_AS_WRITTEN; NOT_FOUND_IN_AVAILABLE_SOURCES]

For v2 onwards, calling tbar invariant under arbitrary projective one-form xi still requires `delta(d epsilon)=3xi`, while for a scalar it is `d(delta epsilon)`. Nonclosed xi cannot be realized. The affine shift also exits the compatible antisymmetric spin-connection space at fixed metric. These problems in A01 do not disappear because a Stueckelberg action is named. The spectator v² term is not automatically invariant under an arbitrary local shift of v either; its claimed shift symmetry epsilon→epsilon+constant is weaker than that projective transformation. No full invariance calculation is supplied.

The optional addition `(alpha/4)epsilon NY` is **not** a constant multiple of an exact form. With Q=e_A wedge T^A,

```
epsilon*dQ = d(epsilon Q) − d epsilon wedge Q.
```

The second term needs evaluation and does not follow to be a boundary merely because NY=dQ. Under strictly pure trace Q=0 it vanishes, but then it cannot create the missing nonzero C2 bulk coefficient. On unrestricted torsion it changes the variation and cannot be used as an innocuous all-purpose improvement without calculation. v2 A66, v3 Eq103 and Symmetry A71 provide no route-specific nonzero CS+ normalization derivation.

## H11 — normalization left inconsistent [VERSION_MISMATCH]

v2 A65 and v3 Eq102 define IT=−Pi[T²]/4 and immediately assert IT=−6eta²Sigma. Their own main contraction is Pi[T²]=6eta²Sigma, which implies−3eta²Sigma/2 instead. Symmetry A70 retains that quarter factor despite main Eq12 switching to IT=−Pi[T²]. Hence this appendix is not a convention-matched derivation of the journal's main invariant coefficient.

## Recovery adjudication

Present: explicit supplementary action, an algebraic trace equation, exact multiplier enforcement, and a classical hard-limit condition. Missing: correct generic finite solution in the source, matched original couplings/eta, complete action variation, consistent arbitrary projective gauge action, and a derived low-energy tensor operator. Classification: **PARTIAL_DERIVATION with explicit CONTRADICTED_AS_WRITTEN finite-mass inference**, not recovery of the missing full C1/C3 proof. The same error is propagated v2→v3→Symmetry rather than resolved by an earlier version.
''')
