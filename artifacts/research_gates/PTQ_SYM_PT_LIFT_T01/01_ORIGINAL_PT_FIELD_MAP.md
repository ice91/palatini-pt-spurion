# Gate 1 — original PT field map

**T0 for the exact original field/composite-action domain: insufficiently specified, with separate explicit inconsistencies in required published identities/domain claims.** The coordinate involution and a conditional abstract scalar involution are well defined at a narrower level. They do not constitute a verified lift on the full original variational domain. No unspecified field law is supplied as a repair.

## Authority and original domain

Primary: `refs/symmetry-18-00170-v2.pdf`, pp.4–8 §§2–3, pp.10–12 §4, pp.28–30 App.A; page numbers are PDF file pages and match printed pagination. Eq.(3) p.4 defines T^A=de^A+ω^A_B∧e^B and R^AB=dω^AB+ω^A_C∧ω^CB. The domain is an oriented four-dimensional Lorentzian Riemann–Cartan geometry with nondegenerate coframe and **metric-compatible antisymmetric spin connection** ω^AB=−ω^BA. The varied fields are e,ω, with fixed scalar spurion ε at the declared order; Eq.(13) p.8 is the original action skeleton. Neither an unrestricted metric-affine connection nor a separately varied compensating one-form belongs to this original problem.

P.6 Eq.(6) defines Π=(I+PT)/2. P.28 Eq.(A1) additionally appends “real scalar”. P.29 App.A.3 explicitly **assumes/asserts** that PT is an involutive automorphism of the local functional algebra. This is not an independent component-by-component verification. Table 1 p.6 explicitly says its representative P/T labels are used only to enforce scalar selection and do not impose microscopic constraints on every e,ω component.

## Published transformation ledger

“NS” below means not specified by the published law, not that no mathematical completion could ever be designed. Bundle/coordinate pullback formulas noted as requirements are typechecking necessities, not adopted new transformations.

| Object / source | Domain → required codomain | Published pullback/coordinate rule | Internal-index rule | Linearity and background action | Composite compatibility / involution result |
|---|---|---|---|---|---|
| Coordinates, p.28 App.A.1 | Local spacetime patch → itself or its transformed patch | P:(t,x)→(t,−x); T:(t,x)→(−t,x); θ=PT:x→−x | None | Real coordinate maps | θ²=I independently verified; det dP=det dT=−1, det dθ=+1. Does not establish a global involution on arbitrary curved/domain data |
| Numeric complex coefficients, p.6 §2.7, A1/A6 | Complex coefficients → conjugates | T includes complex conjugation; coefficient pullback follows position dependence only once specified | NS for quaternionic units | Anti-complex-linear, hence real-linear | Ordinary complex conjugation squares to I. No quaternionic component/product law is given in Symmetry |
| Coframe e^A_μ, Eq.(3)/Table 1 | Nondegenerate Lorentz coframes → same bundle/domain over θD | Table P+,T+; full coordinate/component vs intrinsic-form convention NS | Local Lorentz matrix/bundle lift and its square NS | Inherits anti-linearity only after a full rule is specified; a real coframe subdomain can be contemplated but is not the missing lift | g=ηee must transform consistently. Representative sign squared is not a field-level involution proof |
| Connection ω^AB_μ, Eq.(3)/Table 1 | Affine space of Lorentz connections → Lorentz connections | Table P+,T−; complete connection pullback NS | Internal Lorentz action and any inhomogeneous frame term NS | Background/real structure action NS | Antisymmetry must survive; curvature and torsion must be induced by this same map. Not independently checked |
| Torsion T^A_μν, Eq.(3)/Table 1 | Bundle-valued 2-forms induced from e,ω → same type induced from transformed e,ω | Table P+,T−; indexed Jacobian factors NS | Must agree with coframe/connection bundle action; NS | Representative parity only | UT must equal T[Ue,Uω] with correct d/wedge action. Not proved by a separate sign label |
| Curvature R^AB_μν, Eq.(3)/Table 1 | Adjoint-valued 2-forms → corresponding transformed curvature | Table P+,T+; complete pullback NS | Induced from transformed connection; NS | Representative parity only | UR must equal R[Uω]. Not proved by the table |
| Metric g, A7 p.29; A11 p.30 | Real Lorentz metrics induced from e → same signature metrics | A11 calls metric PT-even; indexed pullback NS | η must be preserved by the coframe lift | A background fixed at g requires θ*g=g (up to declared bundle conventions), rather than a sign alone | Induced transformation is necessary; no original background isometry/invariant field-domain theorem is supplied |
| Scalar ε, pp.5–6 A6/Table 1 | Prescribed scalar profiles → same fixed profile or a declared profile orbit | Table P+,T− | No scalar internal-index action specified | Not varied; p.29 A6 proof also says PT does not act on the fixed spurion | Fixing a variational parameter is distinct from invariance under PT. Its orbit vs fixed-profile convention is unresolved |
| dε / ∂με, Table 1 | Exact 1-forms derived from ε → exact 1-forms from Uε | Table P−,T+ with “vector-like under P”; full time/spatial component rule NS | Induced, not independent | Background derivative must transform by the chain rule | U(dε)=d(Uε) for an ordinary natural differential lift. Assigning independent uniform component signs can violate it (diagnostic below) |
| Scalar/tensor products, p.29 A6 prose | Local functionals/tensors → same algebra/type | Calls PT an automorphism, commuting with derivative/index operations under A2 | Must preserve contracting tensors and order | Real-linear automorphism assumed; anti-complex-linear coefficients | Product compatibility is assumed, not checked from field maps. Does not make Π multiplicative (report 02) |
| Wedge and exterior derivative, Eq.(3), App.A | Ω^p×Ω^q→Ω^(p+q), d:Ω^p→Ω^(p+1) | No separate full field-dependent map; natural pullback would preserve wedge order and commute with d | Bundle action must be compatible with covariant differential | Complex conjugation is compatible with real d; quaternionic order reversal cannot be presumed absent/present | Cartan identities constrain any possible table interpretation; no complete lift supplied |
| Hodge star, A2 p.5, A11 p.30 | Ω^p(g)→Ω^(4−p)(g) | [PT,*]=0 asserted from orientation/metric posture | Bundle pairing must be invariant | A naturality statement transports g to its pullback; fixed-* commutation additionally needs compatible fixed g | Assumption, not an independent check for arbitrary backgrounds. δ* is a different operation; A9 has the prior audited conformal defect |
| Volume, scalar densities, A1 p.5; A1/A8 pp.28–29 | Weight-one densities/top forms → same type on transformed domain | det dθ=+1; integral invariance assumed. No complete complex-density action separately constructed | Levi-Civita object Table P−,T−, hence PT+ | Complex conjugation vs real-part restriction must be tracked | Orientation result verified for θ; measure/background invariance is extra. Density products have weights, not a closed weight-one algebra |
| Fixed background and boundary, A1/A4/A6; A12 p.30 | Admissible domain with prescribed profile, data/fall-offs → same variational problem | PT-invariant domains, compact-support/no-flux or AF/FRW fall-offs assumed | Boundary normals and frame data must transform consistently; NS | No general fixed profile/state/boundary transport law | θD=D alone is insufficient for the full boundary-value problem to be invariant. Not independently established |

For an internal bundle lift, squaring the combined pullback/internal/conjugation action has its own cocycle condition; merely squaring each ± label never verifies it. The source supplies no such condition on its actual e,ω background domain. Likewise a projector of scalar functionals is not itself a projector onto a space of geometric configurations.

## Exact bounded consistency diagnostics

These tests keep original definitions. The two table readings below are **diagnostics of underspecification**, not new adopted transformations. Because the source does not choose between them, neither is promoted into a proof that every possible PT extension is impossible.

### 1. Derivative signs: uniform complete-component reading fails

Suppose Table 1's combined signs are read as complete uniform component laws in the fixed chart, evaluated at θx. They give Uε(x)=−overline(ε(−x)) and U(∂με)(x)=−overline((∂με)(−x)). The chain rule instead gives

    ∂μ[−overline(ε(−x))]=+overline((∂με)(−x)).

For the real prescribed ε=t, Uε=t, so d(Uε)=dt, while the uniform tabulated derivative sign gives −dt. This is a direct contradiction for that reading, including on a PT-invariant scalar background. No choice of couplings repairs the chain rule. Table 1's “vector-like under P” and its representative status may mean the signs are not intended as uniform complete component laws; that leaves the exact rule unspecified, rather than validating this reading.

The geometric e,ω,T,R component signs themselves need not fail Cartan identities on this same chart reading: one derivative flips a sign under θ, and ωω has two ω factors. Therefore we do **not** claim a torsion contradiction merely from observing opposite e and ω labels in a component table.

### 2. Intrinsic signs multiplying natural form pullbacks also cannot be silently assumed

If instead the representative signs are taken as intrinsic factors multiplying ordinary form pullback, the putative e+,ω− rule would yield

    T[Ue,Uω]=θ*(overline(de)−overline(ω∧e)),
    assigned UT=−θ*(overline(de)+overline(ω∧e)).

Their difference is 2θ*overline(de), generically nonzero. At ω=0 take e^0=dt, e^1=exp(t)dx, e^2=dy, e^3=dz. This is a nondegenerate Lorentz coframe with compatible ω=0 and de^1≠0, so the residual does not vanish. Similarly, that reading of ω− and curvature+ gives a −2θ*dω discrepancy in R; a single Lorentz-generator connection ω^01=t dx has nonzero dω and zero ω∧ω. These are conditional kinematic diagnostics, not proposed source fields or solved physical backgrounds.

A different degree/index/bundle convention could alter these intrinsic signs. The point is that the paper does not specify that convention; neither problematic reading may be silently installed as the original field map. No new completion is supplied here.

### 3. Independent original gauge-domain failures

These are not mere table-reading ambiguities. Eq.(4)–(5) p.4 requires δ(dε)=3ξ for an arbitrary one-form ξ. Necessarily dδ(dε)=0, hence dξ=0. The allowed arbitrary ξ=x dy has dξ=dx∧dy≠0. A scalar cannot implement that transformation. Separately, with the declared fixed metric, the affine trace shift changes metric compatibility by δ(∇νgαβ)=−2ξνgαβ; it is not a transformation within the original Lorentz-connection domain. These are previously established counterexamples, re-derived as differential/type identities here, not a new full-action calculation.

Consequently a claim about PT on the exact **projectively reduced original** variational domain is blocked independently of how the discrete table is read. No unrestricted metric-affine geometry, independent one-form compensator, exact-ξ restriction or historical multiplier is substituted to make it work.

### 4. Evenness and realness are different

For a scalar pullback/conjugation model U f(x)=overline(f(−x)), the complex function f=i x¹ is U-even while not pointwise real. Eq.(6) does not universally guarantee pointwise real output merely because U is anti-linear. Eq.(A1)'s extra real-part restriction must either define a smaller domain from the outset or be specified as an additional compatible operation. This is not an argument against real original scalar inputs; it identifies the missing complex-domain qualification.

## What is and is not verified

Independently verified here: θ²=I, orientation determinant, complex conjugation squared, generic formal projector/product algebra under stated hypotheses, chain-rule diagnostics and the fixed projective-domain obstructions. The source's asserted scalar automorphism may be used conditionally. **U_PT²=I on the complete original field/composite variational domain is not independently established.** Once that link fails, claims of a sourced projected gravitational dynamics/Schur operator are stopped; the separate Particles source analysis remains permitted.

Prior evidence: `artifacts/audits/SYM_FULL_A01/03_C1_INDEPENDENT_AUDIT.md` §§C1.9–C1.11; `artifacts/audits/SYM_HIST_P01/02_C1_ACTION_EVOLUTION.md` H03–H06 and `03_STUECKELBERG_APPENDIX_AUDIT.md`; G01 `01_PT_PROJECTOR_TYPECHECK.md`, `02_THREE_CHAIN_EQUIVALENCE_LADDER.md`, `04_CAUSALITY_NONCIRCULARITY_GATE.md`, `06_FINAL_RESEARCH_DECISION.md`. Existing C1/C2/C3 defects remain fixed boundaries; this task does not rewrite them.
