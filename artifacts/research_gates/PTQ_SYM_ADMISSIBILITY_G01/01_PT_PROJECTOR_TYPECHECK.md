# Gate A — PT projector typecheck

**Target verdict: PROJECTOR_NOT_OPERATIONAL_FOR_TARGET / BLOCKED_BY_UNDEFINED_OPERATION.** A formal scalar-sector projector formula exists. It is not a constructed projection on the complete physical variational/principal-symbol domain needed for the proposed causal claim. This distinction is essential: “no formula” would misstate the source.

## What the primary source actually defines

Symmetry p.6 §2.7 Eq.(6) gives Π[O]=(O+PT[O])/2 and prints Π²=Π; time reversal is anti-unitary, conjugating scalar coefficients. P.28 App.A.1 Eq.(A1) adds a restriction to “real scalar”. P.29 App.A.3 calls PT an involutive automorphism of the local functional algebra. Thus involutivity is asserted, not absent; its lift to the claimed fields/composites is not constructed component by component. A1–A6 (p.5) supply domain/measure invariance, orientation/Hodge compatibility, variation commutation, boundary admissibility, NY boundary treatment and a fixed spurion truncation. Theorem 1 (p.6) is the immediate even/odd selection identity.

| Required action | Source and exact content | Typecheck / limit |
|---|---|---|
| Coordinates, P and T | p.28 App.A.1: P:(t,x)→(t,−x); T:(t,x)→(−t,x) | Combined Jacobian determinant +1; pullback on indexed dynamical fields still needs specifying |
| Complex coefficients | p.6 §2.7; p.28 A1; p.29 A6 | Conjugation included in T; PT and Π are generally real-linear, not complex-linear |
| Quaternionic components | No quaternionic component law in §§2–4/App.A–B | Optional PTQ narrative p.5 §2.4 is explicitly not a premise; no i,j,k conjugation/order/action domain is supplied |
| Coframe | p.6 Table 1: P +, T + | Representative parity label; not a complete indexed pullback/internal-frame prescription |
| Metric | p.30 A11 prose: metric PT-even; p.29 A7: g=ηee | Evenness asserted, no full background orbit/invariant perturbation space specified |
| Connection | p.6 Table 1: P +, T − | Representative label; no connection pullback plus internal transformation built to verify compatibility with Eq.(3) |
| Torsion | p.4 Eq.(3) T=de+ω∧e; p.6 Table 1: P +, T − | Its composite transformation must follow from the same e,ω,d action; table alone does not establish that extension |
| Curvature | Eq.(3); Table 1 P +, T + | Composite compatibility not explicitly verified |
| Spurion and gradient | Table 1: ε P+,T−; ∂ε P−,T+; A6 fixed background | “Not varied” does not mean PT-invariant. App.A.3 p.29 says PT does not act on fixed spurion; a background-orbit convention reconciling these statements is missing |
| Hodge star | A2 p.5, p.30 A11: [PT,*]=0 | An assumption on metric/orientation/domain; not a quaternionic Hodge construction |
| Volume/integration | A1 p.5: ∫X=∫PT[X]; App.A.1 orientation; A8 p.29 metric variation of volume and epsilon | Requires PT-invariant integration data. Scalar vs density and density weights of products must be tracked |
| Boundary | A4–A5 pp.5–6; App.A.5 p.30 A12; App.D p.36 compact-support/FRW/AF variations | Flux/variational restrictions, not a PT-derived classification of all boundaries or charges |

Table 1 says it is only for scalar selection and does not impose microscopic constraints on all components. It cannot simultaneously be promoted into a complete field map without supplying what the paper deliberately leaves open.

## Domain/codomain and algebraic properties

| Operator/domain pair | Codomain and linearity | Idempotence / invariant image | Composites and action-level status |
|---|---|---|---|
| Formal Π on a specified PT-stable real vector space D of scalar densities with real-linear involution U | D⁺={O:UO=O}; real-linear | (1+U)²/4=(1+U)/2 if U²=1; UΠ=Π | A valid conditional algebraic projector. A1 permits action restriction; neither arbitrary U-even functions nor pullbacks are automatically pointwise real |
| Same on complex density space with anti-linear U | Real fixed subspace, usually not a complex subspace | Same proof over R | U-even is not synonymous with pointwise real: conjugation plus x→−x can fix complex-valued functions. Additional “take real part” operation in A1 must be specified and commute with U for the combined projector |
| Published scalar bookkeeping Eq.(6), A1–A6, Table 1 | Claimed real PT-even scalar sector | Printed/asserted and conditionally valid using involution; not verified on complete C1/C2/C3 field algebra | Formal symmetry-sector selector plus assumed constraints; not an independently executable map from full action to physical tensor operator |
| e,ω,T,metric field spaces | No Π field-space codomain is defined | Not established / not applicable to scalar-only formula | No field-level geometric uniqueness follows from scalar idempotence |
| C2 determinants / NY-Hodge composites | Claimed scalar bulk line after C1 | Missing full composite/background transformation; determinant expansion additionally branch-dependent | Real coefficients and an imposed C1 substitution do not complete PT typechecking |
| Physical Hilbert states/observables | No Hilbert projector/metric/representation supplied in Symmetry | Not applicable | Cannot import a state-space projection from another action/framework |
| Physical tensor principal symbols | No map from specified PT domain to constraint-reduced operator supplied | Not established | PROJECTOR_NOT_OPERATIONAL_FOR_TARGET |

No fresh physical projector is defined here. The abstract involution calculation is a conditional identity, not validation of the missing lift.

## Multiplication and operation order

Π is **not generally multiplicative**, even if U is an automorphism. Symmetry p.28 Proposition A1 Eq.(A2) asserts ∫Π[X]Π[Y]=∫Π[XY]; p.30 App.A.6 invokes it for C2. This assertion effectively requires an unjustified product rule after integration. Even in the favorable real scalar reading with a fixed measure, U:x→−x on [−1,1], X=Y=x gives ΠX=ΠY=0 but Π(XY)=x², integral 2/3. For literal density×density an additional weight/type specification is required before that equation can be interpreted. An automorphism property of U never implies that of (1+U)/2. This is a re-evaluation of the already reported counterexample, not a new experiment.

| Operation compared with Π | Evidence / result |
|---|---|
| Variation | A3 p.5, Lemma 1 p.6, A6 p.29 assert/prove it using an equivariant, field-independent involution on fields and variations at fixed admissible background. Conditional under those hypotheses, not supplied for all target composites. Conjugation commutes with real variations; the variation direction must transform too |
| Hodge variation | A9–A10 p.30 contain a separate formal defect recorded in SYM_FULL_A01/03_C1_INDEPENDENT_AUDIT.md §C1.11: with its definition of H·α the coefficient is half the needed index term; fixed 2-form conformal check in 4D fails. [PT,*]=0 is not a proof about δ* |
| Integration by parts | A1,A2,A4,A5 allow conditional transport/discarding of actual exact terms. No general theorem upgrades arbitrary asserted improvements to equal bulk densities |
| Auxiliary elimination | Needs a differentiable stationary solution of the full projected action, correct boundary category and compatible symmetry action. C1 of Eq.(13) is not demonstrated to supply it. No general Π–elimination commutation theorem is established |
| Projective gauge quotient | Eq.(4) arbitrary ξ conflicts with scalar-gradient integrability in Eq.(5), and with the metric-compatible field space. No valid common domain or commuting square with Π is supplied |
| Hilbert-sector restriction | A state projector acts on states/represented operators with declared complex structure/metric; distinct from scalar density averaging |
| Schur/Feshbach reduction | Eliminates complementary coupled variables, e.g. A−BD⁻¹C when D invertible; strict projection retains A. Equality needs additional conditions, not idempotence |

Projective quotient identifies gauge orbits; scalar averaging selects even densities; Hilbert projection retains a state subspace; Schur reduction incorporates eliminated couplings. None is interchangeable with the others.

Audit authority: `artifacts/audits/SYM_FULL_A01/03_C1_INDEPENDENT_AUDIT.md` §§C1.7–C1.11; `artifacts/audits/SYM_HIST_P01/02_C1_ACTION_EVOLUTION.md`; `artifacts/audits/MAG_AUX_INDEPENDENT_A01/02_CLAIM_ADJUDICATION.md` M08–M10. Gate A stop applies: subsequent reports only classify existing evidence and bounded written algebra, with no new action, PT lift or physical derivation.
