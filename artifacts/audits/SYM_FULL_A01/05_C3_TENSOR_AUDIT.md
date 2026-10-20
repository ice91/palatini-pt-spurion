# C3 tensor audit

## Physical verdict

**The published nontrivial unique locking theorem is contradicted as written. The original physical K(w),G(w) are not derived in the supplied sources. PT-specific or universal posture-wide tensor luminality is not established.** A narrower, ordinary Einstein–Hilbert principal-symbol argument survives under explicitly nondegenerate algebraic auxiliary elimination. That is independent of the defective locking proof and does not certify the whole published posture or its full DoF count.

Sources: PDF pp.16–20 Eqs30–40, pp.36–39 A42–A55; original gw modules, mixing script/config/meta and C3 tests. No repository function that sets K=G or cT=1 is used as an oracle for the independent physical result.

## E1 — rank and nullspace of the published system

Write Eq33 as `M w=0`, with a square2x2 matrix. The paper p.17 and Proposition A4 p.37 assert detM≠0 generically. Then `M^−1` exists and necessarily `w=0`: rank2, nullity0. A one-dimensional nonzero solution ray requires **rank1, detM=0**, with a separate nonzero-rank condition; rank0 instead leaves all weights free.

The paper's Eq34 equality of two row ratios is precisely a rank1 consistency condition, the opposite of its non-collinearity claim. The contradiction occurs before any K,G computation. It is a mathematical failure, not repaired by choosing a rank1 numerical example. Merely changing a determinant sign/inequality in one sentence would not establish the missing ADM coefficients or correct the subsequent boundary inference.

No evaluated symbolic matrix exists in the PDF. A47–A51 name c0,cH,cY, and TablesA1/A2 repeat the names as entries. Thus the determinant polynomial in actual physical backgrounds cannot be reconstructed from those tables. The asserted `Delta ~ alpha0 eta^4 Sigma² k²`, alpha0≠0, has no supplied evaluated coefficient. The sketch that background-dependent ratios cannot be proportional also does not follow: two proportional rows may have ratios that vary with the background. Constancy in H,Y is not necessary for proportionality of the two rows at each background.

## E2 — exact configured example and rounding distinction

JSON matrix:

```
M = [[0.75, −0.50],
     [0.20, −0.13333333333333333]].
```

Its natural *intended* rational reconstruction is

```
M_Q = [[3/4, −1/2],
       [1/5, −2/15]],
row2 = (4/15)row1,
detM_Q=0, rank=1,
kerM_Q=span{(2/3,1)}=span{(2,3)}.
```

This proves only existence and uniqueness **up to normalization for this chosen matrix**. Both0.75 and0.20 are manually supplied coefficients. None is evaluated from background quantities or Eq13.

If the decimal strings are interpreted literally as exact rationals, the last entry is `−13333333333333333/10^17`. The exact determinant is **1/(4×10^17)**, rank2, nullspace{0}. The script parses binary floats and accepts ratios within tolerance; its two ratios round to the same floating value. The tiny decimal defect is ordinary finite-precision representation of the intended rank1 example; it is not the central physics objection. Exact intended arithmetic and literal-file arithmetic are reported separately to avoid conflating a numerical tolerance with a physical rank theorem.

## E3–E4 — provenance of every matrix element

|Element|Original value|Published provenance|Repository path|Status|
|---|---:|---|---|---|
|mu_ROD(N)|.75|A43/A48 define extraction only; TableA1 placeholder|mixing_matrix_FRW.json N.ROD → load_mu_from_json|SOURCE_NOT_DERIVED|
|mu_CM(N)|−.50|A43/A49; TableA1 placeholder|N.CM|SOURCE_NOT_DERIVED|
|mu_ROD(divN)|.20|A44/A50; TableA1 placeholder|divN.ROD|SOURCE_NOT_DERIVED|
|mu_CM(divN)|−.13333333333333333|A44/A51; TableA1 placeholder|divN.CM|SOURCE_NOT_DERIVED|

`mixing_matrix_extract.py` says explicitly: “physics-side extraction should produce the JSON. Here we only stage, validate, and log.” The appendix reproducibility note instead says this script represents a symbolic ADM expansion. The actual implementation has no metric perturbation, action density, differentiation, tensor projection, lapse, shift or background-solving operation. It loads JSON, divides coefficients, writes CSV/meta/MD5.

The2:3 ratio is computed arithmetically from **manually supplied inputs**; it is not itself hardcoded as a numeric literal in the ratio routine. The underlying action-to-input step is absent. The stored metadata repeats the supplied matrix and computed ratios, with no symbolic provenance. No evidence of the author's motivation for those numbers is available; the audit does not claim to know how they were selected outside the supplied repository.

## E5, E9 and E11 — what independent K,G result is possible

The following scoped derivation is allowed by the existing Eq13 structure, without modifying it.

1. For a metric-compatible connection, write omega=omega_LC+K. The EH scalar is R_LC plus a divergence and algebraic quadratic torsion terms. The source's remaining LO densities are algebraic torsion/spurion-gradient contractions.
2. **If all full-action algebraic blocks are nondegenerate**, their genuine stationary solution can be substituted. The result is EH plus a local function of g and fixed d epsilon, with no new derivatives of g. A sign-fixed local function such as |Sigma| remains algebraic in the metric as well.
3. Such algebraic background terms have no two-derivative metric principal contribution. They can change background equations, lower-derivative terms, consistency conditions and constraints; they do not add a new TT kinetic/gradient principal term to EH.
4. Independently expand the original EH ADM density `M_Pl² N sqrt(gamma)[R3+KijK^ij−Ktrace²]/2` on a flat admissible patch. Put one TT polarization `gamma=diag(1+u q,1−u q,1)`, q=q(t,z). Direct Christoffel contraction gives

```
[sqrt(gamma) R3]_(u²) = 2q q_zz + (3/2)q_z²
                     = −(1/2)q_z² + partial_z(2q q_z),
[sqrt(gamma)(KijK^ij−Ktrace²)]_(u²) = (1/2)q_dot².
L_TT,principal = (M_Pl²/4)(q_dot²−q_z²).
```

This calculation is executed symbolically in `test_tensor_speed_from_EH_ADM_principal_terms`, including the spatial Christoffel/Ricci contraction; it does not read any repository K,G provider. It yields equal nonzero kinetic and gradient coefficients and a luminal massless tensor principal cone. In the paper's two-polarization normalization this is the usual K=G=1 after factoring M_Pl²/8, but the equality here is **derived from EH**, not assigned as an oracle.

At general smooth backgrounds this is a local principal-part statement with regular EH coefficient, consistent tensor identification, and no other derivative metric interactions. It does not supply the missing original weighted-route ADM expansion, solve a background, or validate every original A1–A6 configuration. It gives no special role to PT, C2 or a unique weight ratio.

Minimal sufficient conditions for `cT²=G/K=1` are a valid physical tensor quadratic operator, K=G and K≠0 (K>0 for healthy positive kinetic energy), with no unaccounted principal mixing modifying the eigenmode after constraints. Constant GR normalization and locking uniqueness are not logically required. If K=G=0, division is meaningless and there is no propagating tensor kinetic term to certify.

## E6 — no TT–nonTT mixing is not a universal physical necessity

For a quadratic block action

```
L=(1/2)h A h + h B Phi + (1/2)Phi C Phi,
```

when auxiliary C is invertible, eliminating Phi gives physical kernel `A_eff=A−B C^−1 B^T`. Nonzero B in a chosen variable representation does not itself imply an unphysical theory or a non-spin2 propagating eigenstate. A consistent action with B=0 still need not have equal time/spatial tensor coefficients. Absence of that mixing is neither a general necessary condition for luminality nor sufficient for it. Mixing removal may be a useful variable/gauge convention, provided the constraints and matter coupling are tracked.

There is an additional source-level obstruction in the configured “FRW” interpretation: on a homogeneous isotropic background with epsilon=epsilon(t), any rotation-invariant scalar lapse/longitudinal-shift bilinear with hTT contracts it with combinations of delta_ij and k_i k_j. Both contractions vanish by the trace and transversality conditions. Hence those scalar–TT mixings are absent by the FRW symmetry for every weight in such a covariant scalar action. A nonzeroFRW matrix requires an explicit additional tensor structure or a different precisely defined mixing operator, neither supplied in A43–A44 or the JSON. This observation does not invent replacement coefficients; it identifies why the claimed scalar–TT bilinear must be explicitly defined before “extraction” has meaning.

## E7 — physically equivalent boundary representatives cannot determine bulk locking

Suppose the paper's C2 premise were true: `L1=B+div J1`, `L2=B+div J2`. With constant route weights and A4,

```
w1 L1+w2 L2 = (w1+w2)B + div(w1 J1+w2 J2).
```

At fixed total weight, the physical bulk Hessian and principal equations are independent of the ratio. Boundary/canonical representatives can redistribute displayed terms but cannot select a distinct physical bulk wave speed when all boundary variations are declared irrelevant. Therefore the narrative that two *bulk-equivalent* actions have physically different bulk TT–constraint mixing selectable by their weights needs a calculation distinguishing representation dependence from genuine dynamics. None is provided. This is a second logical obstruction even if the determinant contradiction were removed. The actual C2 densities also fail their asserted equality, as report04 demonstrates.

## E8 — total divergence does not prove equal coefficients

Equations35/38/A53 equate a background coefficient `K(w)−G(w)` with the divergence of a current A54–A55 **quadratic in h**. Under h→r h the right side scales as r² while the left side does not; the displayed objects are not of the same perturbative degree.

Even if one had a well-defined background identity `K−G=div J`, a boundary integral of that coefficient alone does not remove `(K−G)dot h²` or `(K−G)(grad h)²` from the quadratic action. Integration by parts differentiates the h factors and generally gives bulk terms. Exact counterexample: a function f with f(0)=f(1)=0 has integral f'=0, but for h=t²(1−t)³ and f=t(1−t), `integral f' (h')²` is nonzero. Compact variation boundary conditions do not turn this weighted product into a boundary.

A valid result would require a complete **quadratic density identity**, equality of reduced actions up to legitimate boundaries, and then equality of their principal symbols. These are distinct from an integral identity for a coefficient. A52 defines a current using unspecified variational momenta, but K,G themselves are not evaluated. A54–A55 state representatives without showing the required full density relation. Thus the inference “divergence → K=G → physical speed1” is invalid as written, independently of missing numerical coefficients.

## E10 — three different propositions

|Proposition|Verdict at supplied-source scope|
|---|---|
|EXISTENCE_OF_LUMINAL_BRANCH|CONDITIONAL: ordinary EH principal branch survives if valid nondegenerate algebraic elimination and consistent background/tensor reduction apply; original locked branch not derived|
|UNIQUENESS_OF_LOCKING|CONTRADICTED_AS_WRITTEN for detM≠0 nontrivial theorem; DEMONSTRATION_ONLY for intended rank1 JSON example|
|UNIVERSAL_LUMINALITY_IN_POSTURE|UNVERIFIED: posture is internally problematic, full coefficient/constraint chain absent; no universal PT-specific result established|

Failure of uniqueness does not by itself refute a luminal branch. Conversely a luminal EH branch does not validate the published symmetry-selection mechanism.

## E12 — constrained-system DoF evidence

Section7.2 p.20 asserts primary/secondary GR constraints and their first-class algebra, then evaluates `(12−2*4)/2=2`. The arithmetic is correct **if** the four first-class spatial-metric constraints are established after eliminating all auxiliaries; the source supplies no Poisson brackets, Dirac matrix, rank analysis, reduced Hamiltonian or constraint preservation calculation. A trace-lock multiplier Lambda_mu appears in this narrative but not in Eq13 with a complete multiplier action.

An external fixed spurion is not automatically harmless for first-class diffeomorphism constraints: covariant spurion bookkeeping and an actual gauge symmetry at fixed external profile must be distinguished. For a timelike fixed gradient, a reduced `sqrt(-g) C Sigma` term depends nonlinearly on lapse (proportional to1/N in a homogeneous slicing). A claimed GR constraint algebra cannot simply be copied without checking that dependence and the background consistency/Noether identities. This observation does not assert an inevitable extra propagating mode; it identifies missing evidence.

`principal_eigs(locked=True)` allocates n zeros and sets the first two entries to1. `principal_symbol_eigs` returns two chosen effective numbers `[cT²,Z]` for coefficient dictionaries. The actual figure adapter selects `hamiltonian_eigs` and generates a synthetic unlocked n=2 array, `[1.0016,1.002]` at the paper configuration. It never diagonalizes a physical matrix. The original stored `deg_eigvals.csv` already contains exactly two positive numbers. Its “two modes” caption is consequently not a physical DoF count. Even the rank of a genuine velocity Hessian alone would be insufficient without the constraint algebra.

## E13 — complete action-to-figure provenance

|Stage|What the repository does|Classification|
|---|---|---|
|Physical action Eq13 / route densities|No actual connection+metric variation implementation|Missing derivation|
|C1 map|T_vec=alpha*d_eps, pure-trace constructor|Supplied alpha, imposed ansatz|
|C2 coefficients|.125 constants or same JSON values|Hardcoded/configured target|
|ADM mixing|Four JSON entries|Configured; SOURCE_NOT_DERIVED|
|Locked ratio|Two ordinary divisions; require their tolerance agreement|Calculated from assumed inputs|
|locking.apply|Adds `locked=True`|Boolean marker|
|locking.apply_locking|Adjusts one chosen effective coefficient using L=(.25,−.5,.125)|Manual model constraint; weights not derived|
|K,G notebook API|locked returns(1,1); unlocked fixes K=1|Hardcoded|
|cT dispersion|lock=True returns ones; unlocked tanh² deviation of amplitude.02*seps_scale|Hardcoded/demo parameterization|
|Effective test action|cT²=1+L.c; Z=1+.10|grad_eps2|+.05|box_eps||Chosen effective formulas|
|Heatmap|cT=1+.02(X+Y)²; overlaid y=−x|Chosen curve; not derived from mixing2:3|
|Waveform|GR=sin(k*t), model=sin(k*t), k default.05|Identical imposed sinusoids; no propagation solver|
|DoF figure|Two synthetic entries via alias; direct locked API fills38 zeros+two1s|Hardcoded shape/values|
|CSV/PDF/MD5|Deterministic data formatting plus plot rendering/metadata|L5 reproduction/integrity|

The heatmap's line w_CM=−w_ROD also has a different sign/ratio from the positive2:3 configured matrix ray. Its axes are even labelled scale(s),k by the script despite the paper calling them route weights. There is no mapping establishing equivalence of those parameterizations.

No source-permitted complete ADM reconstruction can be finished from the placeholder tables and absent symbolic densities. The bounded audit records this obstruction and stops, rather than choosing new coefficients, backgrounds or a corrected K,G model.
