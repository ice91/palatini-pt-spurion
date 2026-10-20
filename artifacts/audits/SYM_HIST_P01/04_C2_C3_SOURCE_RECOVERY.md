# C2/C3 historical source recovery and NLO provenance

## H12 — original route densities across the four PDFs [PARTIAL_DERIVATION; VERSION_MISMATCH]

These are the **written actions**, not their asserted target reductions. Let x=lambda tau, sigma=sgn(Sigma), n_mu=(partial_mu epsilon)/sqrt(|Sigma|). The following table preserves the sources' placement of sigma.

|Object|v1|v2|v3|Symmetry|Actual change?|
|---|---|---|---|---|---|
|ROD density|file p.15 Eq25; p.34 A29–A31|file p.15 Eq34; p.36 A30–A32|file p.11 Eq19; p.31 Eq67–69|p.14 Eq26; p.34 A31–A33|Same sqrt(-g)[sqrt det(I+(2/3)x mathcalT)−1]|
|mathcalT|file p.8 Eq8; p.14 Eq22; p.34 notation|file p.9 Eq15; p.15 Eq31; p.36 notation|file p.30 App.B/C notation|p.33 App.C notation|All write **sigma*(n^mu n_nu−delta^mu_nu/4)**, not sigma*n^mu n_nu−delta/4|
|CM metric/density|file p.15 Eq26; p.35 A32|file p.15 Eq35; p.36 A33|file p.11 §5.1.2; p.31 Eq70|p.14 §5.1.2; p.34 A34|Same gtilde=g+(2/3)x n n; L_CM=sqrt(-gtilde)−sqrt(-g)|
|CS+ prescription|file p.14 NY/projector propositions; p.15 Eq27; p.35 A33|file p.15 Propositions7–8, Eq36; p.36 A34|file p.11 §5.1.3; p.31 Eq71|p.14 §5.1.3; p.34 A35|Same NY identity→star→scalar projection prescription and asserted nonzero target; no explicit independent nonzero shadow density|
|Common target|A_star=lambda²/8 multiplying sigma IT|Same|Same|Same|Narrative/equation labels change; coefficient claim not action derivation|
|IT convention|−Pi[T²]/4|Same|Same|Main changes to−Pi[T²], but G retains quarter|Actual invariant normalization changes; see02|

The v1 operational discussion(file p.28 §11.4) also drops the2/3 in an informal DBI/CM description while the operative Eq25/26 and AppendixC retain it. No alternate complete action with corresponding coefficient recomputation is supplied there. It is a local wording/formula inconsistency, not evidence of a recovered compensating transformation.

The Stueckelberg mass term and optional epsilon NY action are **new supplementary action terms from v2 onward**, discussed in03. They are not an explicit replacement of the original ROD, CM or CS+ route density. They cannot be silently used to make the earlier three routes equivalent.

## H13 — literal sign placement narrows the previously verified ROD calculation [CONTRADICTED_AS_WRITTEN]

The rendered pages confirm that sigma multiplies the whole parenthesis. With n²=sigma,

```
Tr mathcalT = sigma*(sigma−1) = 1−sigma,
Tr(mathcalT²) = 5/4−sigma/2.
```

Thus the printed definition has Tr=0 and Tr²=3/4 on a spacelike-gradient patch(sigma=+1), but **Tr=2 and Tr²=7/4 on a timelike-gradient patch(sigma=−1)**. The claimed traceless normalization is not valid on both signs, despite the stated sign-compensated scope and homogeneous FRW example.

Direct determinants of the **literal printed** matrix give

```
spacelike ROD: L/sqrt(-g) = −x²/12 + O(x³),
timelike ROD: L/sqrt(-g) = 2x/3 + x²/36 + O(x³).
```

For the actual CM shift, the determinant lemma gives in either sign

```
L_CM/sqrt(-g)=sqrt[1+(2/3)sigma x]−1
            =sigma*x/3−x²/18+O(x³).
```

No one of these expansions licenses equating the two original Jacobians. On sigma=+1 alone, the mismatch−1/12 versus−1/18 already persists. For old IT=−Pi[T²]/4, even that spacelike ROD raw coefficient converts to lambda²/2 multiplying sigma IT using the old version's own consistent pure-trace contraction; the claimed lambda²/8 imports the inconsistent−6eta² relation. This is a check on the old normalization, not a proposed coefficient repair.

**Impact on A01:** A01 independently verified the determinant series under the imposed abstract conditions Tr mathcalT=0, Tr(mathcalT²)=3/4, using the canonical traceless matrix. The historical literal-equation comparison shows that this conditional calculation must **not** be described as validation of the printed timelike matrix definition. This new report records that additional scope correction without editing A01. It strengthens the unresolved-action finding; it does not recover C2. No corrected sign placement is substituted into the source.

## H14 — CS+ is not recovered by an earlier NY/Hodge definition [NOT_FOUND_IN_AVAILABLE_SOURCES; CONTRADICTED_AS_WRITTEN]

All four AppendixC versions start from the ordinary NY identity and assert the same nonzero target. Main text and AppendixC do not always use the same sign when rearranging the curvature term; no version calculates a distinct nonzero operator to bridge that inconsistency.

For the given pure-trace torsion T^A proportional to e^A wedge d epsilon,

```
e_A wedge T^A=0,   T_A wedge T^A=0,   NY=0.
```

Consequently applying an ordinary star to the zero4-form or the scalar PT projector still gives zero. The earlier versions do not define the additional nonzero PT/Hodge operator missing from Symmetry. A common improvement current is selected by convention in all appendicesC, not derived from each original action's difference. v1 also names a CS+ FRW current in §11.3/file pp.27–28, but it is not tied by an operator variation to a nonzero bulk coefficient and does not supply the missing density.

The optional `(alpha/4)epsilon NY` in AppendixG is separately written from v2 onwards. Its weighted-boundary issue and vanishing under strict pure trace are in03. It is not a recovered independent CS+ route giving lambda²/8.

## H15 — all four physical ADM coefficients [NOT_FOUND_IN_AVAILABLE_SOURCES]

|Version|Extraction rules and expansion labels actually present|Coefficient table actually present|Missing calculation|
|---|---|---|---|
|v1|file pp.36–37 A40–A45 and D.12–D.15|file p.37 TablesA1,A2: cells c_ROD/CM,0/H/Upsilon|No evaluated c values, route Hessians, or full-action coefficient substitution|
|v2|file p.38 A41–A46; file p.39 D.12–D.15|file p.39 TablesA1,A2: same placeholder cells|Same|
|v3|file p.33 Eq78–83 and D.12–D.15|file p.35 Tables5,6: same placeholder cells|Same|
|Symmetry|pp.36–37 A42–A51|p.37 TablesA1,A2: same placeholder cells, Y notation|Same|

The PDFs give instructions for second variation and parameterize mu=c0+cH*H+cUpsilon*Upsilon. Those are definitions/recipes, not the completed values of mu_ROD(N), mu_CM(N), mu_ROD(divN), mu_CM(divN). No earlier table supplies the numbers absent in Symmetry. The non-collinearity proofs still appeal to unspecified coefficients, and all four versions repeat the incompatible detM!=0 plus nonzero1D homogeneous-kernel inference.

Read-only Git materially resolves **numerical provenance**, not physics:

1. `1dfaf83279c378e1f8321661113d48e26af0c475` introduced a mixing config with a three-point H/Upsilon grid, fixed trace eta=1 and TrT2=.75, symbolic:false. It includes // comments, so is not strict JSON. At that commit the named extractor script was empty. It is an unexecuted recipe within available evidence, not coefficient data from a completed fit.
2. `b24e203034c419b7a2cde2f8a8c4c823be58b669` first gives the script a body: JSON reader→CSV, explicitly saying a compute path could be added later.
3. `925736072ac3f9e89ca50c8dafa6b9c029ddb008` replaces the recipe by `[[.75,−.50],[.20,+.35]]`. Its exact determinant is29/80, with rank2 and no nonzero kernel.
4. `55f14b84668d9397ac0efb3578b3c4676e6ce96a` changes only the final supplied coefficient to−.13333333333333333 and adds ratio-checking/meta logic. The intended rational matrix has rank1, giving the ray(2,3). Neither source patch derives the revised entry from ADM. The script says physics extraction should produce JSON elsewhere; no such producer is in the accessible history.

Classification of this Git sequence: **CONFIGURED_DEMONSTRATION**. It establishes direct input editing and staged checking, not the author's motive for the edit or any calculation outside the archive. Exact patches are retained in `evidence/history_configs__coeffs__mixing_matrix_FRW.json.txt` and the mixing-script history. No input values were changed by this audit.

## H16 — physical K(w),G(w) [NOT_FOUND_IN_AVAILABLE_SOURCES; ASSUMPTION_CHANGED]

|Version|Source statements|What is still absent|
|---|---|---|
|v1|file p.17 §6.2 explicitly says “We also impose(L2) exact luminality K(w)=G(w)”; pp.20–21 Eqs42–46 then invoke coefficient divergence and impose GR normalization|No independently evaluated constraint-reduced K and G|
|v2|file p.18 §6.2 repeats explicit imposedL2; pp.20–21 Eqs51–55 repeat divergence/current/normalized action|Same|
|v3|file pp.13–16 Eqs23–33 relabels equality as emergence from unique locking|No new evaluated Hessians/auxiliary reduction replacing imposedL2|
|Symmetry|pp.16–19 Eqs30–40 calls it structural proof|Same|

A statement that an expansion has been performed, a schematic quadratic action with unknown K,G,M,P, or a requirement K=G is not an evaluation. The named currents do not fix this: coefficient/divergence and perturbative-degree issues from A01 are already present in earlier versions. No source evaluates the reduced tensor kernel after solving constraints, e.g. the actual contribution of a mixing block through auxiliary elimination. Boundary-equivalent representative actions also cannot acquire ratio-dependent bulk principal symbols at fixed total weight when A4 declares their boundary variations irrelevant.

All7 historical `quadratic_action.py` blobs were inspected. The earliest public coefficient model chooses fixed L weights(.25,−.5,.125), cT²=1+L.c and an assigned positive Z. Later versions return ones when locked and use chosen tanh² offsets otherwise. The notebook helper returning(1,1) is introduced at commit75ab0c6f5adf530df1c47bfb61d525c4a9defbcc. No historical alternative derives these constants by differentiating the original action. This is **CONFIGURED_DEMONSTRATION**, not a recovered physical tensor theorem.

## H17 — Poisson/Dirac algebra [PARTIAL_DERIVATION; CONTRADICTED_AS_WRITTEN; NOT_FOUND_IN_AVAILABLE_SOURCES]

- v1 file p.12 calls multiplier enforcement first class; file p.21 §7.3 asserts no new canonical pairs and exactly two DoF. No bracket matrix is given.
- v2 file p.21 §7.3 is more detailed: primary momenta pi_N,pi_i,p_Lambda; asserted secondaries H,H_i and algebraic torsion conditions; conditional20−2*8 count. This is a recovered **partial constraint inventory**. It still does not display the physical Hamiltonian, original full constraint functions, their nonzero brackets or closure/preservation calculation.
- v2 Lemma11 on that page calls the pair `(p_Lambda, chi=t−3eta v)` second class. On its stated independent t,Lambda variables, `{p_Lambda,chi}=0`; both self brackets vanish. That displayed pair is not a nonsingular second-class bracket matrix. A complete auxiliary constraint set can involve more equations/momenta, but the source does not supply it, and none is invented here. The same section's takeaway that primaries remain first class also needs the actual reduction, not labels.
- v3 file p.17 §7.2 and Symmetry p.20 §7.2 assert the standard GR first-class algebra after eliminating auxiliaries, but give no full bracket calculation. Fixed external epsilon does not make that identification automatic.

All4 historical degeneracy source blobs either choose two effective eigenvalues or manufacture a synthetic zero/positive spectrum. A two-entry diagonal model or two preset ones is not a physical constrained-Hamiltonian analysis. No complete physical Poisson/Dirac algebra is recovered from any PDF, LaTeX asset or notebook.

## H18 — NLO basis and TT calculation [PARTIAL_DERIVATION; CONFIGURED_DEMONSTRATION; NOT_FOUND_IN_AVAILABLE_SOURCES]

|Version|Actual NLO source|Status|
|---|---|---|
|v1|file p.23 Eq50: b/Lambda²*(partial hTT)²*sigma IT; file p.24 Eq52: a directly postulated spatial(grad²hTT)² term; Eq53 quotes slope2 dispersion|Reduced tensor parameterization provided, no covariant operator→TT derivation or complete basis|
|v2|file p.24 Eq59 and Eq61, p.25 Eq62: same two forms/dispersion, with new AppendixG motivation|Same missing reduction; finite Stueckelberg mistake does not supply the NLO operator|
|v3|file pp.19–20 §8.2 Eq36: asserts one operator and directly quotes omega²=k²(1+b k²/Lambda²); footnote says other terms lockstep/boundary|No explicit covariant NLO basis or operator-by-operator TT calculation|
|Symmetry|p.9 Table2; pp.46–49 new AppendixH A72–A93 attempts operator-by-operator classification|This detailed Weyl²-based claim appears here among the supplied versions, not as a completed older derivation. A01's curvature-identity/off-shell objections remain.|

Within a fixed spurion background, the first older form `(partial h)²*IT` has only two h derivatives; counting background derivatives in IT does not independently give an extra tensor k² in cT². The second form directly specifies a four-spatial-derivative tensor term, so a mode polynomial can be written once it is assumed. It does not derive its covariant origin, uniqueness or Wilson coefficient. With the **displayed positive b sign** added to the usual positive kinetic/negative spatial EH action, that isolated term gives omega²=k²−b k⁴/Lambda², not the same-b plus sign quoted in the next equation. Redefining b could change notation but is not performed in this audit.

The single historical `gw/nlo.py` blob (a9236f9892128baa81773f652b7eaede9927d464, first introduced at90495ef560638471c384a56e5999fb17cca8728b) states it is a parameterization and sets DeltaK=a k²/Lambda², DeltaG=(a+b)k²/Lambda². The figure script and notebook plot a chosen k² curve. No hidden older source derives the paper's Weyl² operator reduction. Recovering an explicit reduced ansatz is only a **PARTIAL_DERIVATION of its assumed model**, not proof of the physical source claim.

## H19 — version-matching notebooks and LaTeX [VERSION_MISMATCH; NOT_FOUND_IN_AVAILABLE_SOURCES]

Nine unique.ipynb blobs across the72 commits were decoded, along with their paired.py scripts. All saved outputs are empty and execution counts null. The four notebook topics remain sanity K=G assignment, route-independent symbolic helper, programmed flux convergence and chosen NLO offsets. A symbolic simplification of two calls returning the same preset formula is not an original action reduction. No version-matching executed notebook, ADM action variation, physical constraint algebra or Stueckelberg coefficient derivation was found.

The same last2025 Git tip84501317edf5003e72cf54818075e4319bf9dcf9 is the latest local commit before each of the three preprint posting dates. That temporal coincidence is not a release/convention match; no tag/source map ties the PDFs' differing IT definitions or new appendix material to a distinct code version. References to Supp.R or a “single symbolic pipeline” are not recovered archive objects. The only historical TikZ files are diagrams; no manuscript LaTeX source is accessible in these trees.

## Effect on A01

Recovered: earlier imposed constraints, explicit imposed luminalityL2, supplementary finite-mass equation, historical input edits, raw ROD expansion steps and a partial constraint inventory. Not recovered: any of the four full physical ADM entries, independent physical K,G, complete constraint algebra, a nonzero independently defined CS+ density, or a source-derived NLO tensor reduction. The literal timelike mathcalT issue adds a scope correction to the earlier conditional ROD check. The missing physical chain is confirmed, not completed.
