# Gate 3 — Schur compatibility, with projectors kept distinct

**NO_SOURCED_GRAVITATIONAL_SCHUR_OPERATOR.** Symmetry does not supply a field-level P,Q decomposition and a complementary differential block/inverse from which the requested gravitational Schur operator could be evaluated. Its schematic ADM mixing is not such a completed object. No block is constructed here.

A local exact copy of **Particles 2026, 9, 93**, DOI **10.3390/particles9030093**, was found and read: `/Users/rocky/Downloads/particles-09-00093.pdf`. Its title is *Projected Scalar Green Functions and Schur Reduction of PT-Symmetric Quaternionic Principal-Symbol Deformations*. It is the publisher-paginated 18-page article; title/DOI/journal identity were checked from the document. SHA256 `dd41e88ebd180f9b73312da95d981280aacc0166ccfb515bd0054785b4ba54f2`. The `(1)` download is byte-identical. Two `particles-4522596...pdf` files have different hashes and were not merged into this source. No download, archive copy or manuscript substitution occurred.

## Three different mathematical types

| Object | Domain and scalar structure | What it does | What it does not establish |
|---|---|---|---|
| Symmetry Π_PT=(I+U_PT)/2 | Scalar-density/local-functional space; real-linear for anti-linear PT, assuming a valid lift | Average a density with its declared PT image | Not a complex-linear state projector, not a physical complementary-field split, not block elimination |
| Particles P_Θ=(I+Θ)/2 | Complex auxiliary carrier Γ(E_scalar⊗C²); Θ=diag(1,−1), complex-linear | Retain the + carrier line; Q_Θ=I−P_Θ | Does not make that complex line real; a real scalar slice is separately selected. Not microscopic anti-linear PT, not automatically orthogonal in G_full |
| Schur/Feshbach map D→D_eff | Block differential operators with compatible domains and a prescribed inverse on the complementary block | Incorporate complementary propagation/coupling before eliminating it | Not idempotent averaging, not a projector on density products, not automatically equal to strict restriction |

Particles pp.4–6 Eqs.(12)–(24) define these distinctions explicitly. P.7 Eq.(30)–(31) notes that G_full generally does not commute with Θ when b≠0; an algebraic grading projector must not be relabeled the physical-metric orthogonal projector. A scalar grading of an operator algebra can also be defined by A→ΘAΘ; it is yet another map, with different domain from the carrier projector P_Θ.

## Exact general block statement and domains

Let E=E_P⊕E_Q and P,Q be genuine complementary projectors preserving an appropriate common operator domain. Write A=PDP|E_P, B=PDQ, C=QDP, F=QDQ|E_Q. A source-prescribed inverse R=F⁻¹ must exist on the required source range, with range in the domain needed by B. Products and boundary/state prescription must be consistent. Solving

    Au+Bv=j_P,
    Cu+Fv=j_Q

gives v=R(j_Q−Cu) and

    (A−BRC)u=j_P−BR j_Q.

Thus D_strict=A and D_eff=A−BRC. Equality requires the composite BRC to vanish on the declared domain (or be beyond declared accuracy). Both cross blocks zero is sufficient, not necessary. Even if operators agree, source matching must be checked; elimination can transform the retained source. No new gravity source protocol is inferred from this generic equation.

If a real-linear involution J acts on E, P=(I+J)/2, and D and its domain commute with J, then cross blocks vanish over R. If J is anti-complex-linear, its fixed/anti-fixed spaces are real, not generally complex Hilbert subspaces; it cannot silently serve as Particles' complex-linear Θ. For a different anti-linear symmetry U, a sufficient covariance statement for the effective block requires: U transports or preserves P,Q and domains; UDU⁻¹ obeys the declared block transformation; and URU⁻¹ is the **same prescribed inverse**. Then transformation of BRC follows by ordered composition. These are conditional algebraic requirements, not supplied facts about Symmetry.

## Particles: source-defined grading-odd mixing

P.3 Eq.(4)–(7) defines an auxiliary quaternionic-valued principal-symbol deformation on a **fixed real Lorentzian metric**. It is not a quaternionic spacetime metric. P.4 Eqs.(8)–(13) gives

    δG_H^{μν}=E^{μν}e₂,
    W=ρ(e₂)=[[0,1],[-1,0]],  Θ=diag(1,−1),
    W†=−W,  ΘWΘ=−W,  P_Θ W P_Θ=0.

Both off-diagonal blocks of W are nonzero. Therefore the deformation is Θ-odd and is deleted by **strict** retained projection at first order. This does not mean it is odd under microscopic PT: those are different transformations.

P.8 Eqs.(34)–(40) chooses a differential representative, rather than only naming a symbol:

    B=B_sym=∇μ(E^{μν}∇ν),
    D=[[D++, +B],[-B, D−−]].

E is real symmetric. With compact support or admitted boundary conditions, B is formally symmetric by Eqs.(36)–(37); essential self-adjointness/universal domains are not claimed. Coefficient functions do not commute through derivatives. ΘDΘ flips the mixing, so **[D,Θ] is generally nonzero** when B≠0. The full carrier is not sector-preserving merely because PDP is an operator on the retained subspace (p.6 Eq.(21) discussion).

P.6 Eqs.(22)–(24), Proposition 3 and p.9 Eq.(42) then yield, where the common prescribed inverse exists,

    D_strict=D++,
    D_eff,+=D++−(+B)(D−−)⁻¹_F(−B)
           =D+++B(D−−)⁻¹_F B.

The plus sign follows from the off-block signs and was independently rechecked here in the noncommuting free-word calculation in report 02. No commutation of B with the inverse is needed. The retained block of the **full inverse** is the inverse of this Schur complement, not generally the inverse of D++. The source's frozen symbol p.9 Eq.(43) is a+b²/d versus a (d≠0); if b²/d≠0 these operators already differ at symbol level. No arbitrary numerical values are needed to show the difference.

## Microscopic PT compatibility is an additional, conditional statement

Particles p.5 Eqs.(15)–(17) distinguishes microscopic anti-linear PT, the spacetime involution θ, and complex-linear Θ. It prescribes tensor pullback, complex/quaternionic conjugation at the stated local level, PT i PT⁻¹=−i, PT W PT⁻¹=−W, and the odd transformed scalar symbol b_metric(θx,k_θ)=−b_metric(x,k), with b_metric=E^{μν}kμkν. It does **not** equate anti-linear PT with Θ or prove that all curved backgrounds possess the required isometry.

P.9 Proposition 4 Eq.(41) assumes θ*g=g, a boundary/state prescription respecting the declared transformation, and

    U B U⁻¹=−B,
    U W U⁻¹=−W,
    U D−− U⁻¹=D−−.

With the additional **prescription-level inverse covariance** stated in its proof,

    U(D−−)⁻¹_F U⁻¹=(D−−)⁻¹_F,

one obtains ordered identities

    U(BW)U⁻¹=(−B)(−W)=BW,
    U[B(D−−)⁻¹_F B]U⁻¹=(−B)(D−−)⁻¹_F(−B)
                       =B(D−−)⁻¹_F B.

The microscopic PT-even correction can therefore re-enter at second order even though the first-order Θ-odd mixing was deleted by strict P_Θ projection. This is a sourced conditional compatibility relation; it is not equality of the two projectors. To extend covariance to the entire effective operator and retained state space also requires the stated compatible action on the retained line/diagonal block and domains. Eq.(41) proves the parity of the correction under its hypotheses, not construction of a full gravity lift or a universal global PT realization.

**Inverse caveat:** invariance of a differential expression is not sufficient to fix a Green inverse. Anti-linearity conjugates i0; time reversal may also transport boundary/state data. For example, if one merely conjugates (F+i0)⁻¹ for a real F, the prescription becomes (F−i0)⁻¹. Therefore covariance of a Feynman inverse is not established by UD−−U⁻¹=D−− alone. The source explicitly places state/boundary compatibility among its hypotheses. This gate verifies the conditional implication, not existence of an invariant global Feynman prescription for every background. No particular Green kernel is constructed here.

## Exact Schur versus controlled local expansion

Particles p.9 Eq.(44), p.10 Eqs.(45)–(48) supplies D−−=K_−−M² and a local soft/separation regime. M and the complementary diagonal block are **supplied reduction data**, not derived from δG_H and not necessarily a physical particle mass. With ∥K_−∥/M²≪1 in the stated local-symbol/EFT sense,

    (D−−)⁻¹_F=−M⁻²−M⁻⁴K_−+…,
    δD_eff,+=−M⁻²B²−M⁻⁴BK_−B+…,
    δQ_eff=−[E^{μν}kμkν]²/M²+O(M⁻⁴)

at frozen leading coefficients. This checks the exact plus Schur sign versus negative leading local insertion. For varying E the object is the ordered composition B_sym²; it is not simply multiplication by b_metric². The exact expression remains conditional on existence of the inverse even when the local expansion fails. Real-spectrum/positive-metric inequalities in p.7 Eqs.(29)–(31) and p.8 Eq.(33) are local-symbol restrictions, not a new global causal or stability theorem.

## Source-by-source checklist

| Required check | Original Symmetry | Particles 9,93 |
|---|---|---|
| Genuine P,Q split | No sourced physical complementary field split from Π_PT | Yes: P_Θ,Q_Θ on the explicitly declared complex auxiliary carrier, Eqs.(12),(18),(21) |
| Real vs complex linearity | Π_PT real-linear conditional scalar operation | Θ/P_Θ complex-linear; microscopic PT separately anti-linear |
| D commutes with involution | No original D tied to a verified field involution | D generally does not commute with Θ; microscopic PT covariance requires separate diagonal/background hypotheses |
| Cross blocks | Not evaluated from original action | +B,−B, generally nonzero; Eqs.(35),(40) |
| Complementary inverse | Not sourced | Common state/boundary/i0 inverse **assumed where it exists**, Eqs.(22)–(24); not universally proved |
| Boundary/state transformation | A1–A6 do not provide an operator inverse prescription | Local PT-compatible prescription required by Proposition 4; covariance conditional |
| PT-even retained correction | Cannot derive | B R B even under Eq.(41) and inverse covariance; independently checked ordered signs |
| Strict vs Schur | No comparison available | Different when B R B≠0; exact Eq.(42), local symbol Eq.(43) |
| Causal/observable implication | None from this gate | Local retained scalar insertion derived in source's model; no observed effect or general gravity claim |

The absence conclusion for gravity follows from inspection of the original definitions plus pp.16–19 Eqs.(30)–(40), p.36 A42–A45 and p.37 Tables A1/A2, not merely a keyword search. Those formulas name schematic K,G,M,P and unevaluated coefficients; they do not give the requested explicit complementary differential operator, projector/domain or prescribed inverse. The manuscript's symbolic block labeled P is not P_Θ and is not a density projector.

## Companion limits and stop

`refs/preprints202604.0888.v2.pdf`, PDF pp.6–8 Definition 2, Eqs.(11)–(20), compares complete Hilbert/metric/observable/generator packages with transported preparation and measurement. It cannot identify the above projections or give a gravitational inverse. `refs/preprints202604.1054.v3.pdf`, PDF pp.7–9 §5.6 uses an internal real-linear toy involution, and PDF p.14 §7.3 explicitly distinguishes that map, complex-linear Θ, scalar-density projection and Schur reduction. Those source distinctions support separation, not a bridge to Symmetry.

**T3 is supported as a conditional operator identity in the Particles source's declared common-inverse regime; no realized global inverse/state theorem is newly proved. T3 is not established for Symmetry.** There is no sourced intertwiner from Symmetry's scalar-density involution to this auxiliary carrier grading, no source-derived gravitational B and no valid inference of gravitational re-entry by analogy. Stop at NO_SOURCED_GRAVITATIONAL_SCHUR_OPERATOR.
