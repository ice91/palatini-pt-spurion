# Final claim adjudication — PTQ-SYM-FULL-CHAIN-AUDIT-01

## A. Executive summary

The supplied source chain does **not** establish the published symmetry-enforced C1→C2→C3 mechanism at its stated scope. It contains both explicit mathematical counterexamples and separately missing physical derivations. This is a bounded source audit, not a construction of a corrected theory.

The local50-page v2 PDF is byte-identical to the publisher v2 artifact. The original source HEAD is6b456b89baa866b6dd05bc69a837b98a985a34ad. Existing tests:44 collected, **42 passed+2 skipped**, also42+2 in module-based strict paper mode. A supported full runtime is MISSING_ENVIRONMENT (available scientific environment Python3.9.6 versus required>=3.10); make mixing succeeds, make paper runs eight modules then fails on the last module's Python annotation. No installation or original-file edit was made.

Independent tests: **42 passed,4 NOT_EXECUTABLE**, separate from the original suite. These include exact counterexamples to projector formulas, full-action C1 variation/uniqueness, determinant-route equality, ordinary NY nonzero bulk, locking inference, and NLO reductions. Missing ADM/CS+/constraint sources are not converted into synthetic passes.

## B. Claims independently proved

- Involutive scalar projector idempotence and± selection under its explicit algebraic assumptions.
- Restricted Lorentz-covariant single-vector no-go for the traceless torsion irrep.
- Pure-trace contraction IT=−6eta²Sigma **when pure trace is imposed consistently**.
- The published ROD determinant coefficient lambda²/8 after the declared C1 substitution and on a non-null sign-fixed patch.
- Ordinary NY is an exact form; for pure trace e∧T,T∧T and NY vanish.
- Intended rational demo matrix has rank1, nullspace span(2,3). This is a mathematical fact about supplied numbers, not a physical locking theorem.

## C. Claims contradicted as written

Scalar-gradient compensation of an arbitrary one-form and metric-compatible closure under the printed affine projective transformation fail. Printed irrep projectors/axial normalization and some Appendix A functional identities fail exact tests. The C1 proof drops the curvature torsion terms; alpha1≠0 does not ensure unique full-action elimination, and the printed trace formula/sign/eta relation are inconsistent with the stated conventions/action.

Actual CM and ROD Jacobians differ (quadratic coefficients lambda²/12 versus lambda²/8 multiplying sigma IT). Ordinary pure-trace NY cannot be their common nonzero CS+ bulk. An invertible homogeneous mixing matrix cannot admit the asserted nonzero one-dimensional solution. A coefficient being a divergence does not imply K=G as a physical operator identity. The claimed NLO classification conflicts with curvature identities and off-shell TT Ricci².

These findings distinguish a local false formula from a missing proof; they do not declare every possible model with PT or torsion false.

## D. Claims only demonstrated numerically

C1 alignment/component plots, three equal .125 coefficient paths,2:3 JSON-ratio staging, locked cT arrays, overlaid identical sinusoids, chosen two-eigenvalue spectra, imposed flux convergence and NLO slope2 curves are L4/L5 demonstrations. They are reproducible within the tested environment/path, with final flux regeneration blocked by version compatibility. Numerical equality of manually supplied targets does not create three independent action reductions.

## E. Missing physical derivations

Four principal obstructions are explicitly NOT_EXECUTABLE: evaluated action-derived ADM mixing entries; a defined nonzero CS+ shadow operator; original constraint-reduced K(w),G(w); and physical Hamiltonian/Dirac constraint algebra. TablesA1/A2 contain coefficient labels, not evaluated numbers. `mixing_matrix_extract.py` stages JSON and explicitly expects physics extraction elsewhere. No full source-to-observable chain was found in the other modules or notebooks.

## F. Fundamental root cause

The chain changes evidence type without justification: elementary algebra and configured outputs are promoted to dynamical uniqueness, bulk action equivalence and physical principal-symbol claims. Independent of that provenance gap, several intermediate statements are mathematically false. A single assumed target line is reused where independent route-specific physics was needed. One corrected printed inequality would not supply the missing action variations or resolve the determinant/current/NLO problems.

## G. Does any independent cT=1 proof survive?

**Yes, conditionally and generically:** for an original EH action whose torsion is truly auxiliary and can be eliminated nondegenerately, with no remaining derivative metric interactions beyond EH, the extra algebraic background terms do not alter its tensor principal cone. A direct independent ADM calculation yields equal nonzero kinetic/spatial-gradient coefficients for the EH TT mode. This proves a narrower ordinary luminal principal branch under its stated conditions, not the full published posture, its complete constraint count or its assigned locking mechanism.

Failure of nontrivial locking existence/uniqueness does not by itself imply absence of any luminal branch. Conversely that generic surviving branch cannot validate PT-specific enforcement.

## H. Is PT-specific luminality established?

**No.** Neither a lawful arbitrary projective compensation within the declared connection space nor an action-derived unique physical weight selection is established. The narrower surviving tensor-cone argument follows from the EH principal operator and algebraic auxiliary elimination, without requiring PT selection, three-route equivalence or the configured ratio. Physical novelty specific to PT therefore remains unverified in these sources.

## I. Publication implications — analysis only

The central published proof cannot be cited as independently validated by this repository's44-test historical claim. Reproducibility of figures and checksums supports software/data integrity; it does not settle the action-level mathematical failures. The theorem's published generic-locking reasoning is invalid as written, C2's stated original densities do not agree, and physical coefficients/constraint evidence are absent. Some narrower algebraic and conditional tensor results remain defensible.

No conclusion about author intent, misconduct, all PTQ theories or an appropriate journal procedure is drawn. No publication correction, editorial contact, submission or external finding was made. This report records the evidence rather than developing a new mechanism to rescue it.

## J. Recommended single next action

**Obtain the archived original action-to-ADM derivation for this exact paper version, with full connection variation, evaluated mixing entries, physical K,G and constraint provenance, before treating its central physical theorem as established.** This recommendation is not executed as follow-up research. The audit stops here.

## Claim-by-claim records

All required adjudication fields appear below; machine-readable equivalents are in `evidence/claim_adjudication.json`. Confidence refers to the evidentiary verdict (e.g. high confidence a derivation is absent/invalid), not to a universal assertion of physical nonluminality. Detailed derivations and test names are in03–07. Levels L0–L6 are defined in00/01.


### A1

**CLAIM_ID**: A1  
**PAPER_LOCATION**: p.5 §2.6; App.A pp.28–30  
**SOURCE_CODE**: No domain/measure implementation  
**ORIGINAL_TEST**: None  
**INDEPENDENT_CHECK**: A symmetric interval supplies a consistent involution domain; A1 does not imply A2 product identity  
**ASSUMPTIONS**: Specified PT-invariant integration and allowable background/variations  
**VERDICT**: CONDITIONAL  
**CONFIDENCE**: HIGH  
**IMPACT**: Can be a legitimate assumption; not numerically established for all claimed backgrounds  
**WHAT_WOULD_RESOLVE_IT**: Define/check actual admissible domain and measure  


### A2

**CLAIM_ID**: A2  
**PAPER_LOCATION**: p.5 §2.6; p.30 A11  
**SOURCE_CODE**: palatini_pt/spurion/pt_even.py (no Hodge implementation)  
**ORIGINAL_TEST**: None  
**INDEPENDENT_CHECK**: Orientation-preserving metric-equivariant involution commutes with Hodge; printed variation A9 separately fails conformal check  
**ASSUMPTIONS**: PT preserves orientation and transforms metric consistently  
**VERDICT**: CONDITIONAL  
**CONFIDENCE**: HIGH  
**IMPACT**: A2 can consistently be imposed; does not validate incorrect Hodge variation or arbitrary projective symmetry  
**WHAT_WOULD_RESOLVE_IT**: Explicit field action and consistent Hodge calculus  


### A3

**CLAIM_ID**: A3  
**PAPER_LOCATION**: p.6 Lemma1; p.29 A6  
**SOURCE_CODE**: palatini_pt/spurion/pt_even.py:191 pt_project;:225 project_scalar  
**ORIGINAL_TEST**: No functional variation test  
**INDEPENDENT_CHECK**: Real-linear equivariant involution commutes with real variation on invariant field/background space  
**ASSUMPTIONS**: Field-independent involution on full varied objects; fixed-spurion domain preserved  
**VERDICT**: CONDITIONAL  
**CONFIDENCE**: HIGH  
**IMPACT**: Elementary commutation survives in a defined invariant domain; software does not prove geometric variation  
**WHAT_WOULD_RESOLVE_IT**: Specify admissible background orbit/variations and full projected action calculus  


### A4

**CLAIM_ID**: A4  
**PAPER_LOCATION**: p.5; p.30 A12  
**SOURCE_CODE**: palatini_pt/algebra/ibp.py; equivalence/flux_ratio.py  
**ORIGINAL_TEST**: test_algebra_basis.py; test_flux_ratio.py  
**INDEPENDENT_CHECK**: Compact-support true divergences do not change bulk equations; A38/A40 and coefficient-divergence arguments fail  
**ASSUMPTIONS**: Actual currents are divergences; spatial and temporal boundary variations vanish  
**VERDICT**: CONDITIONAL  
**CONFIDENCE**: HIGH  
**IMPACT**: Valid boundary posture does not make arbitrary asserted route differences into boundaries  
**WHAT_WOULD_RESOLVE_IT**: Verify actual density-level currents and boundary variations  


### A5

**CLAIM_ID**: A5  
**PAPER_LOCATION**: p.5; p.34 C.3  
**SOURCE_CODE**: palatini_pt/equivalence/cspp_chain.py (JSON only)  
**ORIGINAL_TEST**: test_c2_equivalence.py does not test NY  
**INDEPENDENT_CHECK**: Ordinary NY=d(e∧T); pure trace gives e∧T=0 and T∧T=0  
**ASSUMPTIONS**: Smooth forms and admissible boundaries; no defects  
**VERDICT**: PROVED_WITHIN_STATED_DOMAIN  
**CONFIDENCE**: HIGH  
**IMPACT**: Ordinary boundary identity survives; cannot generate asserted nonzero CS+ bulk  
**WHAT_WOULD_RESOLVE_IT**: No additional evidence needed for exact-form identity; nonzero shadow needs its own definition  


### A6

**CLAIM_ID**: A6  
**PAPER_LOCATION**: p.5 §2.6; p.20 §7.2  
**SOURCE_CODE**: palatini_pt/palatini/field_eq.py; spurion/profile.py  
**ORIGINAL_TEST**: C1 tests take external gradient  
**INDEPENDENT_CHECK**: Fixed background/truncation is definable; it does not imply GR first-class constraints  
**ASSUMPTIONS**: Declared fixed spurion and two-derivative LO expansion  
**VERDICT**: CONDITIONAL  
**CONFIDENCE**: HIGH  
**IMPACT**: A valid truncation choice; full diffeomorphism/constraint consequences remain unproved  
**WHAT_WOULD_RESOLVE_IT**: Check fixed-background consistency and constraint preservation in original action  


### THEOREM_1

**CLAIM_ID**: THEOREM_1  
**PAPER_LOCATION**: p.6 Eq6 and Theorem1  
**SOURCE_CODE**: palatini_pt/spurion/pt_even.py:146,:191,:225  
**ORIGINAL_TEST**: No full geometric projector test  
**INDEPENDENT_CHECK**: Exact 2x2 involution projector verifies idempotence and ± eigenspace selection  
**ASSUMPTIONS**: PT²=id, real-linear treatment of conjugation  
**VERDICT**: PROVED_WITHIN_STATED_DOMAIN  
**CONFIDENCE**: HIGH  
**IMPACT**: Elementary scalar selection rule survives; no physical coefficient follows from it alone  
**WHAT_WOULD_RESOLVE_IT**: No missing evidence for this elementary statement  


### APP_A_PRODUCT_IDENTITY

**CLAIM_ID**: APP_A_PRODUCT_IDENTITY  
**PAPER_LOCATION**: p.28 A2  
**SOURCE_CODE**: No implementation  
**ORIGINAL_TEST**: None  
**INDEPENDENT_CHECK**: X=Y=x, PT:x→−x on [-1,1]: LHS0, RHS2/3  
**ASSUMPTIONS**: Real smooth scalars on symmetric domain  
**VERDICT**: CONTRADICTED_AS_WRITTEN  
**CONFIDENCE**: HIGH  
**IMPACT**: Claimed product version of self-adjointness is false  
**WHAT_WOULD_RESOLVE_IT**: A valid original functional identity; no silent substitution in this audit  


### APP_A_HODGE_VARIATION

**CLAIM_ID**: APP_A_HODGE_VARIATION  
**PAPER_LOCATION**: p.30 A9/A10  
**SOURCE_CODE**: No tensor Hodge implementation  
**ORIGINAL_TEST**: None  
**INDEPENDENT_CHECK**: Fixed 2-form under conformal scaling has delta star=0; A9 gives nonzero2u  
**ASSUMPTIONS**: 4D Lorentz metric, displayed H.alpha definition  
**VERDICT**: CONTRADICTED_AS_WRITTEN  
**CONFIDENCE**: HIGH  
**IMPACT**: Formal calculus appendix not validated by proof narrative  
**WHAT_WOULD_RESOLVE_IT**: Consistent derivation of original Hodge variation  


### C1_IRREP_PROJECTORS

**CLAIM_ID**: C1_IRREP_PROJECTORS  
**PAPER_LOCATION**: pp.31–32 A13–A21  
**SOURCE_CODE**: palatini_pt/palatini/torsion_decomp.py:55,:68,:90,:101,:117  
**ORIGINAL_TEST**: test_torsion_decomp.py:12 (pure trace only)  
**INDEPENDENT_CHECK**: A14/A15 flip extracted irreps and fail idempotence; axial norm −S²/6, not +S²/24; code raises epsilon incompletely  
**ASSUMPTIONS**: Declared trace t=T^a_mu a, axial S=epsilon T, Lorentz signature  
**VERDICT**: CONTRADICTED_AS_WRITTEN  
**CONFIDENCE**: HIGH  
**IMPACT**: General irrep algebra and positivity split not proved; pure-trace software subset can still pass  
**WHAT_WOULD_RESOLVE_IT**: Consistent conventions tested on all irreps rather than only axial-zero examples  


### C1_SINGLE_VECTOR_NO_GO

**CLAIM_ID**: C1_SINGLE_VECTOR_NO_GO  
**PAPER_LOCATION**: pp.11,32–33 PropositionA3  
**SOURCE_CODE**: palatini_pt/algebra/basis.py does not derive representation result  
**ORIGINAL_TEST**: None  
**INDEPENDENT_CHECK**: Only g∧v and epsilon.v seeds; trace and axial constraints force both coefficients zero  
**ASSUMPTIONS**: Lorentz covariance, linear in one vector, no additional independent tensors  
**VERDICT**: PROVED_WITHIN_STATED_DOMAIN  
**CONFIDENCE**: HIGH  
**IMPACT**: Valid restricted algebraic fact; not uniqueness of homogeneous dynamical solutions  
**WHAT_WOULD_RESOLVE_IT**: No missing evidence for restricted no-go  


### C1_PURE_TRACE_FORM

**CLAIM_ID**: C1_PURE_TRACE_FORM  
**PAPER_LOCATION**: p.10 Theorem2; pp.11–12 Eqs20–24  
**SOURCE_CODE**: palatini_pt/palatini/field_eq.py:38; connection.py:35  
**ORIGINAL_TEST**: test_c1_torsion.py:22,:38  
**INDEPENDENT_CHECK**: Full action can give unsourced S=q=0 and trace alignment only with actual nonsingular algebraic blocks  
**ASSUMPTIONS**: No axial/tensor sources; full-action blocks nondegenerate; consistent geometry  
**VERDICT**: CONDITIONAL  
**CONFIDENCE**: HIGH  
**IMPACT**: Pure-trace branch is plausible/derivable on narrower domain, not the unrestricted published uniqueness theorem  
**WHAT_WOULD_RESOLVE_IT**: Complete original connection variation and nondegenerate coefficient domain  


### C1_EXACT_TRACE_MAP_SIGN

**CLAIM_ID**: C1_EXACT_TRACE_MAP_SIGN  
**PAPER_LOCATION**: p.8 Eq10; p.10 Eq16; p.32 A24  
**SOURCE_CODE**: palatini_pt/palatini/connection.py:35 uses opposite inversion sign  
**ORIGINAL_TEST**: Original pure-trace tests do not compare printed trace convention  
**INDEPENDENT_CHECK**: Eq16 traces to −3eta v, opposite to +3eta v in Eq10/21  
**ASSUMPTIONS**: Unit-weight antisymmetrization and stated trace  
**VERDICT**: CONTRADICTED_AS_WRITTEN  
**CONFIDENCE**: HIGH  
**IMPACT**: Printed formula and declared trace lock cannot both hold with those conventions  
**WHAT_WOULD_RESOLVE_IT**: Single consistent original torsion/trace convention  


### C1_UNIQUENESS

**CLAIM_ID**: C1_UNIQUENESS  
**PAPER_LOCATION**: p.10 Theorem2; p.12 alpha1≠0 claim  
**SOURCE_CODE**: palatini_pt/palatini/field_eq.py directly chooses solution  
**ORIGINAL_TEST**: test_c1_torsion.py tests constructed ansatz  
**INDEPENDENT_CHECK**: Explicit EH algebraic trace/axial/q terms admit singular full-action blocks at nonzero alpha1  
**ASSUMPTIONS**: Original arbitrary real coefficients, nonzero EH term  
**VERDICT**: CONTRADICTED_AS_WRITTEN  
**CONFIDENCE**: HIGH  
**IMPACT**: alpha1≠0 is insufficient; theorem exceeds proven parameter domain  
**WHAT_WOULD_RESOLVE_IT**: Original full-action Hessian coefficients and singular-locus analysis  


### C1_ETA_FORMULA_AND_SIGN

**CLAIM_ID**: C1_ETA_FORMULA_AND_SIGN  
**PAPER_LOCATION**: p.12 Eqs21–24  
**SOURCE_CODE**: palatini_pt/palatini/field_eq.py alpha parameter; c1_pure_trace.py angle=0  
**ORIGINAL_TEST**: C1 tests choose positive alpha  
**INDEPENDENT_CHECK**: Full trace denominator includes EH contribution; eta>0 imported from C3 rather than independently derived  
**ASSUMPTIONS**: Full Eq13, not Eq20 alone  
**VERDICT**: CONTRADICTED_AS_WRITTEN  
**CONFIDENCE**: HIGH  
**IMPACT**: Printed eta relation not full-action result; positive sign not independently fixed  
**WHAT_WOULD_RESOLVE_IT**: Complete connection variation and justified positivity condition without circular C3 use  


### C1_IT_IDENTITY

**CLAIM_ID**: C1_IT_IDENTITY  
**PAPER_LOCATION**: p.10 Eq17; p.12 Eq24; p.33 A29  
**SOURCE_CODE**: palatini_pt/equivalence/coeff_extractor.py reduce_to_IT assumes identity  
**ORIGINAL_TEST**: No original all-action contraction test  
**INDEPENDENT_CHECK**: Independent contractions give T²=2t²/3 and IT=−6eta²Sigma under imposed pure trace  
**ASSUMPTIONS**: Consistent pure-trace ansatz, real eta; sign of trace irrelevant to square  
**VERDICT**: PROVED_WITHIN_STATED_DOMAIN  
**CONFIDENCE**: HIGH  
**IMPACT**: Valid conditional algebraic reduction survives; dynamics of C1 not supplied by it  
**WHAT_WOULD_RESOLVE_IT**: No missing evidence for contraction; physical application inherits C1 conditions  


### PROJECTIVE_COMPENSATOR

**CLAIM_ID**: PROJECTIVE_COMPENSATOR  
**PAPER_LOCATION**: pp.4–5 Eqs4–5  
**SOURCE_CODE**: No actual gauge transformation implementation  
**ORIGINAL_TEST**: None  
**INDEPENDENT_CHECK**: xi=x dy has dxi≠0, so d(delta epsilon)=3xi impossible  
**ASSUMPTIONS**: Published arbitrary projective one-form, scalar compensator  
**VERDICT**: CONTRADICTED_AS_WRITTEN  
**CONFIDENCE**: HIGH  
**IMPACT**: Advertised arbitrary projective gauge redundancy not realized  
**WHAT_WOULD_RESOLVE_IT**: Lawful original transformation for stated arbitrary xi; cannot silently restrict to exact xi  


### PROJECTIVE_METRIC_COMPATIBILITY

**CLAIM_ID**: PROJECTIVE_METRIC_COMPATIBILITY  
**PAPER_LOCATION**: p.4 §§2.1–2.3 Eq4  
**SOURCE_CODE**: palatini_pt/palatini/connection.py assumes metric compatibility  
**ORIGINAL_TEST**: None  
**INDEPENDENT_CHECK**: delta(nabla_nu g_ab)=−2xi_nu g_ab  
**ASSUMPTIONS**: Fixed metric; affine projective trace shift; antisymmetric spin connection  
**VERDICT**: CONTRADICTED_AS_WRITTEN  
**CONFIDENCE**: HIGH  
**IMPACT**: Transformation exits declared variational connection space  
**WHAT_WOULD_RESOLVE_IT**: Actual symmetry action on the originally declared field space  


### C2_ROD

**CLAIM_ID**: C2_ROD  
**PAPER_LOCATION**: p.14 Eq26; p.34 A31–A33  
**SOURCE_CODE**: palatini_pt/equivalence/dbi_chain.py:7 (JSON); __init__.py:31 (fixed provider)  
**ORIGINAL_TEST**: test_c2_equivalence.py fixed public API  
**INDEPENDENT_CHECK**: Exact determinant gives −lambda²tau²/12=(lambda²/8)sigma IT  
**ASSUMPTIONS**: Paper traceless Tcal, small lambda*tau, imposed C1, non-null sign-fixed patch  
**VERDICT**: PROVED_WITHIN_STATED_DOMAIN  
**CONFIDENCE**: HIGH  
**IMPACT**: One algebraic route coefficient survives; code is still L4  
**WHAT_WOULD_RESOLVE_IT**: No missing algebra; physical use requires valid C1/action normalization  


### C2_CM

**CLAIM_ID**: C2_CM  
**PAPER_LOCATION**: p.14 metric shift; p.34 A34  
**SOURCE_CODE**: palatini_pt/equivalence/closed_metric_chain.py:7; __init__.py:35  
**ORIGINAL_TEST**: test_c2_equivalence.py compares chosen constants  
**INDEPENDENT_CHECK**: Matrix determinant lemma gives linear term plus −lambda²tau²/18; A_CM=lambda²/12  
**ASSUMPTIONS**: Exact printed rank-one shift and n²=±1  
**VERDICT**: CONTRADICTED_AS_WRITTEN  
**CONFIDENCE**: HIGH  
**IMPACT**: CM does not match ROD coefficient; A34 uses wrong matrix determinant  
**WHAT_WOULD_RESOLVE_IT**: Source-grounded valid transformation or equality of declared actions; none supplied  


### C2_CS_PLUS

**CLAIM_ID**: C2_CS_PLUS  
**PAPER_LOCATION**: p.14 §5.1.3; p.34 A35  
**SOURCE_CODE**: palatini_pt/equivalence/cspp_chain.py:7; __init__.py:41  
**ORIGINAL_TEST**: test_c2_equivalence.py target comparisons  
**INDEPENDENT_CHECK**: Ordinary NY pure trace zero; no nonzero extra operator/normalization supplied  
**ASSUMPTIONS**: Strict pure-trace C1; ordinary versus additional Hodge operator distinguished  
**VERDICT**: BLOCKED_BY_MISSING_SOURCE  
**CONFIDENCE**: HIGH  
**IMPACT**: Nonzero CS+ route is not an independent derivation; ordinary-NY reading contradicts claimed bulk  
**WHAT_WOULD_RESOLVE_IT**: Exact original nonzero CS+ density, lambda normalization, explicit reduction  


### C2_THREE_ROUTE_EQUIVALENCE

**CLAIM_ID**: C2_THREE_ROUTE_EQUIVALENCE  
**PAPER_LOCATION**: p.14 Proposition1 Eqs27–28; Apps.C  
**SOURCE_CODE**: palatini_pt/equivalence/coeff_extractor.py:25,:32,:45; order2_checker.py:8  
**ORIGINAL_TEST**: test_c2_equivalence.py; test_paper_mode_apis.py  
**INDEPENDENT_CHECK**: ROD/CM coefficients differ; ordinary NY=0; JSONs share assumed target  
**ASSUMPTIONS**: Original densities, no invented subtractions or field redefinitions  
**VERDICT**: CONTRADICTED_AS_WRITTEN  
**CONFIDENCE**: HIGH  
**IMPACT**: Central C2 physical equivalence fails for supplied densities; L4 comparisons remain reproducible  
**WHAT_WOULD_RESOLVE_IT**: Actual route-specific derivations and a legitimate density-level equivalence proof  


### C2_IMPROVEMENTS_AND_FLUX

**CLAIM_ID**: C2_IMPROVEMENTS_AND_FLUX  
**PAPER_LOCATION**: pp.15,35 Eqs29,A36–A41  
**SOURCE_CODE**: palatini_pt/equivalence/flux_ratio.py:20; scripts/fig_flux_ratio.py  
**ORIGINAL_TEST**: test_flux_ratio.py; C2 unlisted-key drop  
**INDEPENDENT_CHECK**: A40 divergence differs from A38 for h=t²+x²; assigning common currents cannot prove bulk equality  
**ASSUMPTIONS**: Actual currents and admissible boundary variations; denominator nonzero for flux ratio  
**VERDICT**: CONTRADICTED_AS_WRITTEN  
**CONFIDENCE**: HIGH  
**IMPACT**: Flux→1 does not validate original actions; plot is a chosen convergence curve  
**WHAT_WOULD_RESOLVE_IT**: Explicit valid route currents/density differences and genuine flux integration  


### C3_NONTRIVIAL_LOCKING_EXISTENCE

**CLAIM_ID**: C3_NONTRIVIAL_LOCKING_EXISTENCE  
**PAPER_LOCATION**: p.17 Eq33; p.37 PropositionA4  
**SOURCE_CODE**: palatini_pt/gw/locking.py; scripts/mixing_matrix_extract.py  
**ORIGINAL_TEST**: test_mixing_matrix_extract.py (configured matrix only)  
**INDEPENDENT_CHECK**: detM≠0 implies kerM=0  
**ASSUMPTIONS**: Published homogeneous square system and nonzero determinant  
**VERDICT**: CONTRADICTED_AS_WRITTEN  
**CONFIDENCE**: HIGH  
**IMPACT**: Claimed nonzero generic locked solution cannot follow from stated condition  
**WHAT_WOULD_RESOLVE_IT**: Original physical matrix with valid nontrivial-kernel domain argument  


### C3_LOCKING_UNIQUENESS

**CLAIM_ID**: C3_LOCKING_UNIQUENESS  
**PAPER_LOCATION**: p.17 Theorem3; p.38 D.6  
**SOURCE_CODE**: palatini_pt/gw/locking.py:61 changes coefficient; locking_curve y=−x  
**ORIGINAL_TEST**: test_c3_tensor.py:86,:150  
**INDEPENDENT_CHECK**: One-dimensional kernel requires rank1, not claimed rank2; physical rank not evaluated  
**ASSUMPTIONS**: Full physical matrix, equivalence of constraints/variables  
**VERDICT**: CONTRADICTED_AS_WRITTEN  
**CONFIDENCE**: HIGH  
**IMPACT**: Unique physical weight selection not established; generic EH luminality does not require it  
**WHAT_WOULD_RESOLVE_IT**: Evaluated original ADM matrix and correct uniqueness proof  


### C3_CONFIGURED_2_TO_3

**CLAIM_ID**: C3_CONFIGURED_2_TO_3  
**PAPER_LOCATION**: mixing artifact / AppendixD reproducibility note  
**SOURCE_CODE**: configs/coeffs/mixing_matrix_FRW.json; scripts/mixing_matrix_extract.py:64,:72,:83  
**ORIGINAL_TEST**: test_mixing_matrix_extract.py:11,:23,:43  
**INDEPENDENT_CHECK**: Intended rational matrix rank1 has kernel(2,3); literal decimal exact rank2 det1/(4*10^17)  
**ASSUMPTIONS**: Chosen matrix and floating tolerance; no physical coefficient provenance  
**VERDICT**: DEMONSTRATION_ONLY  
**CONFIDENCE**: HIGH  
**IMPACT**: Ratio arithmetic is reproducible; no action-derived physical locking  
**WHAT_WOULD_RESOLVE_IT**: Original ADM extraction producing all four JSON entries  


### C3_K_EQUALS_G

**CLAIM_ID**: C3_K_EQUALS_G  
**PAPER_LOCATION**: pp.17,19 Eqs35–39; p.38 A52–A55  
**SOURCE_CODE**: palatini_pt/gw/quadratic_action.py:87 tensor_action_coeffs; :23 cT2_of_k  
**ORIGINAL_TEST**: test_c3_tensor.py; test_paper_mode_apis.py  
**INDEPENDENT_CHECK**: Coefficient divergence is not quadratic-density equivalence; h² current has wrong perturbative degree; original K,G absent  
**ASSUMPTIONS**: Full action and constraint-reduced quadratic operator, K≠0  
**VERDICT**: UNVERIFIED  
**CONFIDENCE**: HIGH  
**IMPACT**: Proof is invalid as written; physical equality on a particular branch may still hold  
**WHAT_WOULD_RESOLVE_IT**: Original evaluated physical K(w),G(w), density-level improvement identity and reduced principal symbol  


### C3_PHYSICAL_CT_EQUALS_1

**CLAIM_ID**: C3_PHYSICAL_CT_EQUALS_1  
**PAPER_LOCATION**: p.17 Theorem3; p.19 Eq40  
**SOURCE_CODE**: palatini_pt/gw/tensor_mode.py:45,:57; quadratic_action.py  
**ORIGINAL_TEST**: test_c3_tensor.py locked arrays; API tests  
**INDEPENDENT_CHECK**: Hardcoded locked ones not used as oracle; independent EH cone survives only conditionally  
**ASSUMPTIONS**: Actual physical tensor eigenmode, full constraints eliminated, no missing derivative interactions  
**VERDICT**: UNVERIFIED  
**CONFIDENCE**: HIGH  
**IMPACT**: Published universal PT-specific luminality not proved; this is not a proof of nonluminality  
**WHAT_WOULD_RESOLVE_IT**: Action-to-coefficient-to-constraint-to-observable derivation for original source  


### C3_TWO_PHYSICAL_DOF

**CLAIM_ID**: C3_TWO_PHYSICAL_DOF  
**PAPER_LOCATION**: p.20 §7.2; PDF Fig10  
**SOURCE_CODE**: palatini_pt/gw/degeneracy.py:13,:29,:43; scripts/fig_c3_degeneracy.py:34  
**ORIGINAL_TEST**: test_c3_tensor.py:118; test_paper_mode_apis.py:15  
**INDEPENDENT_CHECK**: Count conditional on four first-class constraints; figure yields chosen two-entry synthetic array, no brackets  
**ASSUMPTIONS**: Actual primary/secondary constraints and full fixed-spurion algebra  
**VERDICT**: BLOCKED_BY_MISSING_SOURCE  
**CONFIDENCE**: HIGH  
**IMPACT**: Two TT polarizations do not prove full theory has only two physical DoF  
**WHAT_WOULD_RESOLVE_IT**: Original constrained Hamiltonian, bracket algebra, ranks and consistency conditions  


### NLO_OPERATOR_UNIQUENESS

**CLAIM_ID**: NLO_OPERATOR_UNIQUENESS  
**PAPER_LOCATION**: p.9 Table2; pp.47–49 A77–A93  
**SOURCE_CODE**: palatini_pt/gw/nlo.py minimal two-label model  
**ORIGINAL_TEST**: test_nlo.py:10,:19,:28  
**INDEPENDENT_CHECK**: C²=GB+2Ricci²−2R²/3; off-shell TT Ricci²≠0; (nabla Sigma)² has six derivatives  
**ASSUMPTIONS**: Original claimed four-derivative basis and off-shell TT scope  
**VERDICT**: CONTRADICTED_AS_WRITTEN  
**CONFIDENCE**: HIGH  
**IMPACT**: Complete-basis/unique-Weyl physical reasoning not valid at stated generality  
**WHAT_WOULD_RESOLVE_IT**: Consistent original basis quotient and full off-shell TT reduction with branch prescription  


### NLO_SLOPE_2_PHYSICAL_PREDICTION

**CLAIM_ID**: NLO_SLOPE_2_PHYSICAL_PREDICTION  
**PAPER_LOCATION**: p.22 Eq43; p.49 A91  
**SOURCE_CODE**: palatini_pt/gw/nlo.py:65,:93; scripts/fig_nlo_offsets.py:42  
**ORIGINAL_TEST**: test_nlo.py:28 doubles assigned k²  
**INDEPENDENT_CHECK**: Flat covariant curvature-square kernel retains z=0 massless root; b from operator not derived  
**ASSUMPTIONS**: Leading nonzero b, complete covariant reduction and low-energy branch  
**VERDICT**: UNVERIFIED  
**CONFIDENCE**: HIGH  
**IMPACT**: Distinctive physical slope2 prediction not established; does not rule out conditional k² effective forecasts  
**WHAT_WOULD_RESOLVE_IT**: Original operator-to-b derivation including time derivatives, normalization, background and branch  


### NLO_SLOPE_2_DEMO

**CLAIM_ID**: NLO_SLOPE_2_DEMO  
**PAPER_LOCATION**: repo fig8 / notebook30  
**SOURCE_CODE**: palatini_pt/gw/nlo.py:93; scripts/fig_nlo_offsets.py:42; notebooks/30_nlo_offsets.py  
**ORIGINAL_TEST**: test_nlo.py:28  
**INDEPENDENT_CHECK**: Curve proportional to k² has logarithmic slope2 by definition  
**ASSUMPTIONS**: b≠0 and chosen parameterization; perturbative offset  
**VERDICT**: DEMONSTRATION_ONLY  
**CONFIDENCE**: HIGH  
**IMPACT**: Software forecast arithmetic survives; not evidence for physical uniqueness or data detection  
**WHAT_WOULD_RESOLVE_IT**: No missing evidence for demo formula; physical interpretation needs previous record  


### GENERIC_EH_LUMINAL_PRINCIPAL_BRANCH

**CLAIM_ID**: GENERIC_EH_LUMINAL_PRINCIPAL_BRANCH  
**PAPER_LOCATION**: Eq13 structure; independently scoped test  
**SOURCE_CODE**: palatini_pt/gw/quadratic_action.py not used as oracle  
**ORIGINAL_TEST**: Original tests do not derive this branch  
**INDEPENDENT_CHECK**: Direct ADM EH second variation gives equal q_dot² and q_z² coefficients; eliminated algebraic terms add no metric derivative principal part  
**ASSUMPTIONS**: Full-action nondegenerate auxiliary solution, consistent smooth background, regular EH coefficient, no extra derivative metric couplings  
**VERDICT**: CONDITIONAL  
**CONFIDENCE**: HIGH  
**IMPACT**: Independent ordinary luminal tensor principal cone survives; no PT-specific weight-locking novelty  
**WHAT_WOULD_RESOLVE_IT**: Original model must establish these conditions before applying this result to its full posture  
