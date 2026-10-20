# Axioms, assumption budget and exactly two candidates

All items in this ledger are **NEW ASSUMPTIONS / NEW CANDIDATE DEFINITIONS**, unless identified as an existing mathematical identity. No parameter is fitted and no arbitrary function is introduced.

| ID | Mathematical definition and independent motivation | Covariance / PT / dynamics role | Free data introduced | If removed / simpler control |
|---|---|---|---|---|
| A1 | M=R^4, orientation dt∧dx∧dy∧dz; real nondegenerate Lorentz coframes e and real compatible ω^{AB}=−ω^{BA}; both time orientations allowed | Geometric seed and closed pullback domain; no affine projective symmetry | Field configurations, not constitutive functions | Restricting future coframes at fixed time orientation makes θ leave the domain; use paired time-oriented sectors or stop that fixed-domain claim |
| A2 | θx=−x; natural Ue=θ*bar(e), Uω=θ*bar(ω), U on coefficient-valued forms by pullback/conjugation | Explicit involution with no internal field-sign assignments | No tunable parameter | Ordinary diffeomorphism covariance is the real-field control; no special geometric selector |
| A3 | Real constant coefficients; compactly supported variations or genuinely matched boundary data | Makes variational IBP legal; transformed action has transformed domain | At most finitely many coefficients in a chosen operator inventory | Arbitrary fixed complex coefficients or unmatched boundaries need separate transformation; retarded data are not PT-invariant |
| A4 | Candidate 1 action S_EC=(M_P^2/4)∫ε_ABCD e^A∧e^B∧R^{CD}, M_P^2>0 | Standard minimal first-order gravitational kinetic term; no torsion or spurion lock imposed | One normalization M_P^2 | Vacuum Einstein–Cartan control already has exactly this action; zero coefficient would not determine ω |
| B1 | Candidate 2 is a probe doublet Ψ=(φ,χ)^T of two complex scalar fields on the flat torsion-free vacuum of Candidate 1 | Minimal two-sector model capable of actual complementary propagation; not a gravitational tensor model | Two complex dynamical scalars; no background profile | One scalar has no nontrivial Schur complement. Background geometry is fixed for the probe calculation |
| B2 | Θ=diag(1,−1), P=(I+Θ)/2; UΨ(x)=Θ bar(Ψ(−x)) | One internal grading distinguishes sector behavior; anti-linear PT distinct from complex-linear P | Fixed representation, no parameter | U_0=Kθ* is the H0 control. Pure internal linear Z2 would forbid the mass mixing below; generic anti-unitary Z2 reproduces it |
| B3 | Canonical Lorentz-covariant principal operator □I, at most two derivatives and a constant complex symmetric mass matrix | Equal derivative kernels chosen for a minimal calculation; not derived from PT | Before restriction: six real mass entries; after PT: a,b,g real, each mass dimension 2 | General PT permits kinetic mixing too. Omitting it is a substantive restriction. Changing it could change characteristics; current cone no-go is limited to this seed |
| B4 | Solve Θ M* Θ=M in the complex symmetric class, obtaining M=[[a,ig],[ig,b]], D=□I−M | PT imposes an actual coefficient reality/phase relation; real a,b,g not initially restricted to healthy region | Three constants only | Dropping PT permits all complex symmetric entries; a general real/Hermitian control with transported sources reproduces the healthy revised model |
| B5 | Initial S_0=∫[−∂tildeΨ^T∂Ψ−tildeΨ^T MΨ], with Ψ and tildeΨ independent for variation | Minimal variational generator of D, not yet a real/unitary physical action | A formal dual field, removed by the physical-pairing revision | Identifying tildeΨ=Ψ* naively makes the mass action non-real at g≠0. That attempted physical interpretation is rejected |
| R1 | Sole controlled revision: seek constant Hermitian G>0 with M†G=GM, then set tildeΨ^T=Ψ†G; choose a concrete solved witness with G11=G22=1 | Positive-pairing existence is a consistency criterion, not a premise that all parameters are admissible; real physical action then derived | Witness depends on a,b,g, no added free function; general compatible metric is not unique | Positive norm is not guaranteed by PT. At exceptional/broken points no such positive metric exists. Equal diagonal normalization is a chosen witness, not a symmetry prediction |
| B6 | Declare source j and detector via the G pairing; for the displayed response j=e1 J, detector e1†GΨ, e1†Ge1=1 | Gives a precise mathematical response and matched physical preparation | Specified source direction, no fitted function | Bare coordinate inverse is not that response. Transport source and detector under equivalence maps |
| B7 | Retarded inverse for causal sourced response, zero incoming complementary homogeneous solution; also report advanced/PT transport | Physical state/boundary input independent of PT; exact elimination requires the declared inverse | A state/history prescription, no constitutive parameter | General elimination includes complementary homogeneous initial data. PT sends retarded to advanced; a single retarded inverse is not invariant |

## Candidate 1: natural geometric lift

The geometric definitions are those in A1–A4, not the ambiguous historical Table 1. There is no scalar compensator, arbitrary-one-form projective gauge, quaternionic metric or imposed pure-trace torsion. The action-derived vacuum equation sets **all torsion** to zero on a nondegenerate coframe. Flat e^A=dx^A, ω=0 solves both connection and coframe equations.

## Candidate 2: scalar carrier on that vacuum

The NEW initial carrier is not claimed to be a physical positive theory before R1. Its equations and PT symmetry are mathematically meaningful; a physical pairing must still exist. After R1 the explicit physical action is

    S_G=∫d^4x [−∂μΨ† G ∂^μΨ − Ψ†GMΨ
                    − Ψ†G j − j†GΨ].

Signature is (−,+,+,+), □=−∂t²+∇², and z=ω²−|k|². Field dimension is one; a,b,g have mass dimension two and G is dimensionless. The action contains two derivatives and a fixed quadratic mass term, no scalar potential beyond these finitely many masses, interaction or constitutive function. The flat zero-field background Ψ=0 is stationary. If the EC geometry were included as a spectator dynamical seed, scalar stress is quadratic and vanishes on this branch; no coupled gravity fluctuation theorem is claimed.

Define Δ=b−a>0 by relabeling the two diagonal masses, r=2g/Δ, and the solved witness

    G=[[1,−ir],[ir,1]].

The regular healthy region is **derived**, not built into B4: Δ²>4g², tr M>0, det M>0. A simple nonempty sufficient open subdomain is 0<a<b and 0<|g|<Δ/2. Later sections also exhibit a larger exactly healthy isospectral family without requiring individual a>0. At g=0 the sectors decouple; the benign a=b,g=0 degeneracy is distinguished from a nonzero-g exceptional point.

## The assumption that most limits discovery

Canonical Lorentz-invariant □I in B3 already fixes the scalar principal cone. PT constrains lower-order coefficients, so this sprint can discover a consistency domain and nontrivial response but cannot honestly claim that PT generated a new causal cone. This is an explicit scope of the limited no-go, not hidden tuning. The final proposed continuation changes only this one assumption and is not executed here.
