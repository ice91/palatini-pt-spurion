# Two retained maps of the SAME fixed geometric parent

**Decision:** two independently motivated, well-defined PT-compatible retained maps exist, but generically neither defines an autonomous physical subsector. Exact Schur descriptions with the same parent sources/detectors are **SAME_REPRESENTATION_CLASS**. Changing to a map's own preferred source/detector is **DIFFERENT_PREPARATIONS_ONLY**. No distinct admissible universes are established.

Throughout, ℳ²,β,a,b,u, background, boundary/state rules, observable algebra and source functions are unchanged. Use the parent/action and derived K of report 01, with 0<κ<1 and a,b>0, a≠b. No mass, coupling or background is adjusted between P1 and P2.

## 1. Domain, motivation and idempotence

After the exact algebraic connection solve, scalar perturbations are sections with internal positive pairing K. Complexification is only the usual Fourier/Green-function carrier of real fields. The physical quadratic Hessian is D=zK−diag(a,b), mapping fields to duals. Its K-identified equation operator is

    𝒟=K^(−1)D=zI−K^(−1)diag(a,b).

Retained maps below are endomorphisms of this scalar carrier. Thus PDP means P𝒟P after the declared K identification, or the corresponding bilinear compression of D. Applying the same unpaired matrix on both sides of a field-to-dual Hessian would be a type error. Sources transform contragrediently in the action, as calculated below.

Let e1=(1,0)^T, u=(1,1)^T/√2, w=(1,−1)^T/√2, c=(1+κ)/2 and h=(κ−1)/2. Then K=[[c,h],[h,c]], detK=κ.

- **P1: species-line retention.** The line span(e1) is distinguished before diagonalizing the coupled kinetic/mass operator by the original independent species and mass/source labels. Use the K-orthogonal projector onto that line, not an arbitrary Euclidean pairing.
- **P2: torsion-bright retention.** The line span(u) is independently distinguished by the sole geometric current βt·d(u^TΦ). It is selected from the interaction, not from desired poles.

For a nonzero line vector v the K-orthogonal projector is v(v^TK)/(v^TKv). Explicitly,

    P1=e1 e1^T K/c=[[1,h/c],[0,0]],
    P2=u u^T=[[1/2,1/2],[1/2,1/2]],   Qi=I−Pi.             (13)

Direct multiplication gives Pi²=Pi, PiQi=QiPi=0, Pi^TK=KPi. Both have rank one and are distinct. They are real-linear on real fields, complex-linear on the Fourier complexification and commute with U=θ*conjugation. They are **not** field-space anti-linear even averaging (I+U)/2. Their coefficients depend only on the same fixed action's K.

The domain is the reduced stationary-auxiliary submanifold (e,Φ,C*(e,dB)). A lift there maps Φ→PiΦ and reconstructs C* from the projected scalar; squaring this map gives itself. It is not claimed to act as an idempotent symmetry on arbitrary off-shell ω while changing Φ and leaving ω unchanged. Metric/coframe variables are retained unchanged. Neither map is assumed to take a full solution to another solution; closure is tested below.

The constant maps preserve ordinary smooth/Sobolev source domains. Retarded and advanced data are paired by PT; both projector calculations use the same retarded zero-incoming response. Physical real structure is unchanged.

## 2. Canonical retained/complement bases

Define k=√κ>0 and

    n1=e1/√c,
    q1=(−h/(√c k), √c/k)^T,
    n2=u/k,   q2=w,
    Ti=[ni,qi],   Φ=Ti yi.

Both Ti^T K Ti=I, and Pi=Ti diag(1,0)Ti^(−1). The canonical equation operators are Di=Ti^T D Ti=zI−Hi, where

    H1=[[a/c,                 −ah/(c k)],
        [−ah/(c k),  ah²/(κc)+bc/κ]],

    H2=[[s/κ, d/k], [d/k,s]],
    s=(a+b)/2,   d=(a−b)/2.                                (14)

These are congruent descriptions of the same fixed parent Hessian, not two actions with retuned coefficients. The transition O=T2^(−1)T1 is

    O=1/√(1+κ) [[k,1],[1,−k]],
    O^TO=I,   D2=O D1 O^T.                                (15)

This is a regular change of canonical scalar coordinates preserving the physical pairing, while all parent observables are reconstructed by Φ=Ti yi and the same torsion formula (5).

## 3. Dynamical closure: explicit failure and limited no-go

A rank-one K-orthogonal sector is dynamically closed only if [K^(−1)diag(a,b),Pi]=0, equivalently its canonical off-block vanishes. In (14),

    μ1=−ah/(c k),    μ2=d/k.

For a>0, 0<κ<1, a≠b, both are nonzero. Therefore neither P1 nor P2 defines an autonomous one-scalar sector of the parent. In particular the retained subspace does not remain invariant under evolution, even though its map is idempotent and PT-compatible.

**Limited no-go for simultaneous closure of these two motivated maps.** Under a,b>0 and κ>0, μ1=0 forces κ=1 and μ2=0 forces a=b. Thus both maps are simultaneously closed only in the decoupled, mass-degenerate control κ=1,a=b, where any flavor rotation is an ordinary representation/preparation freedom. No nontrivial simultaneous closed-sector claim survives for the working parent.

The fixed parent of course has ordinary mass-normal modes; spectral diagonalization proves positivity and displays those particle modes. It is not introduced as a third projector candidate or interpreted as additional universes. A proof about P1/P2 is not a theorem that no invariant line can ever exist.

## 4. Actual strict and Schur operators

Write Di=[[Ai,Bi],[Bi,Fi]], with Ai=z−αi, Fi=z−δi, Bi=−μi. In each canonical chart the type-correct strict equation compression and exact conditional elimination are

    Di,strict=Ai,
    Ei=Ai−Bi Fi^(−1)Bi
      =z−αi−μi²/(z−δi).                                  (16)

For P2 this is explicitly

    E2=z−s/κ−d²/[κ(z−s)].                                 (17)

For P1 substitute α1=a/c, δ1=ah²/(κc)+bc/κ and μ1=−ah/(c k). Both corrections are nonzero in the working parent. Their numerators satisfy

    Fi Ei=det Di=det D/κ.                                 (18)

The inverse is a common prescribed inverse of the complementary Klein–Gordon block, not a number at its pole. For flat causal response use its retarded Green operator on past-compact sources and the prescribed zero incoming complementary homogeneous state; general initial data add the corresponding homogeneous source. In Fourier formulas use the common retarded boundary value from Imω>0, or work at non-real resolvent frequency. No contour/prescription is changed between maps.

These are exact **equation/Hessian-resolvent** Schur statements. We do not assert that simply inserting a nonsymmetric retarded kernel into a one-field ordinary variational action produces a causal Euler–Lagrange equation; a causal effective action would require appropriate doubled-history machinery. The present calculation instead derives the full local parent equations first and reconstructs their retarded responses.

Microscopic PT transports each retarded Schur inverse to advanced. It does not leave a single retarded inverse fixed. The scalar correction is real/naturally PT-covariant as a transported operator family; this follows from ordinary covariance and the common prescription, not an extra PT coefficient relation.

## 5. Fixed source and detector, including eliminated information

Fix the physical normalized source vector **once**, before either projector is evaluated:

    v=K n1=(c,h)^T/√c,
    J(x)=v j(x),   detector O(x)=v^TΦ(x),
    v^T K^(−1)v=1.                                      (19)

This describes a normalized species-line preparation in the action-derived pairing. The full parent protocol also permits arbitrary fixed test vectors J and detector rows; the equality below holds for their entire internal response matrix, not just this one example.

In canonical chart i, the same source and detector become si=Ti^T v and si^T yi:

    s1=(1,0)^T,
    s2=(k,1)^T/√(1+κ).                                  (20)

P2 therefore has a nonzero complementary source. Deleting it would change the protocol. Solving the actual complementary equation gives reduced source

    si,eff=si,P−Bi Fi^(−1)si,Q.

The full response is

    Ri=si,eff Ei^(−1)si,eff + si,Q² Fi^(−1),              (21)

where the second term is the complementary source/contact contribution. Products are ordered Green-operator compositions with the same state; constant coefficients commute as required in the displayed Fourier formulas.

Direct inversion gives exactly the same result in both charts:

    R_fixed=v^T D^(−1)v
       =[κz−cb−(h²/c)a]/[κz²−c(a+b)z+ab]
       =1/E1.                                            (22)

More strongly, symbolic calculation verifies the complete source/detector matrix identity

    Ti Di^(−1) Ti^T=D^(−1),  i=1,2.                       (23)

It holds for arbitrary independent parent source/detector vectors, and in the retarded domain because each Ti is constant and the same causal boundary prescription is used. The geometric trace response t_μ=(3β/(2ℳ²))∂μ(u^TΦ) is reconstructed from the same D^(−1), and is equally chart independent. Nonzero torsion response is not lost by using either exact effective chart.

If the source contact term in (21) is omitted for P2, the missing expression is explicitly

    1/[(1+κ)(z−s)]≠0.

An independent test checks that this is exactly the error, rather than a numerical mismatch or an unspecified improvement. No false multiplicativity of PT averaging is used. For an order-preserving involution, the actual identity remains P+(XY)=P+X P+Y+P−X P−Y.

## 6. Poles, weights, cones and strict truncations

The common full pole set is λ± of (12). For either intrinsic normalized retained preparation its response has the form

    (z−δi)/[(z−λ−)(z−λ+)],
    w−=(δi−λ−)/(λ+−λ−),
    w+=(λ+−δi)/(λ+−λ−).                                  (24)

Because Hi is symmetric with μi≠0, λ−<δi<λ+, hence both weights are positive and sum to one. The independent calculation verifies w−w+=μi²/(λ+−λ−)². With the **same** physical source (19), P2 must use (21), and its weights equal those of P1 exactly.

If instead one changes the source to v2=K n2 and the detector to n2^TKΦ, its intrinsic bright response is

    R_bright=1/E2=κ(z−s)/detD.

It generally has different weights from (22), although the poles/cones remain the same. This is a different normalized preparation and detector within the same parent algebra. It is not an inequivalent physical theory produced by P2.

Strictly deleting the complementary variable, source and detector gives one-pole approximations Ai^(−1). Under the fixed source (19) the displayed strict responses would be

    R1,strict=1/(z−a/c),
    R2,strict=[κ/(1+κ)]/(z−s/κ).

Their difference is not evidence of fixed-parent physical multiplicity: the discarded complementary dynamics/data were nonzero. Positive kinetic sign in a truncated line does not make that line a consistent autonomous sector of the parent.

All exact descriptions have the same characteristic metric cone, since det principal D=κ z² and regular canonical changes cannot alter it. Positive kinetic/energy domain is the same κ>0,a,b>0. Exact Schur operators retain the two-mode information through memory/inverse kernels; they do not reduce the parent degree count to one merely because one coordinate is displayed.

## 7. Singular charts versus actual constraints

At Fi=0 the pointwise complementary inverse fails. This need not be a physical singularity:

    detD(z=δ1)=−a²(κ−1)²/(κ+1)²≠0,
    detD(z=s)=−(a−b)²/4≠0.                               (25)

For generic working parameters these are chart failures, not extra modes. The complete response can be finite there; the compensating terms in (21) must be retained. Green-operator inverses still require their stated distributional prescription rather than arbitrary evaluation of a reciprocal.

A genuinely different issue is the parameter boundary κ=0. Analyze it directly in unnormalized B=u^TΦ and D_dark=w^TΦ, without dividing by √κ. The equations are

    −s B−d D_dark=J_B,
    (□−s)D_dark−d B=J_D.

Since s>0,

    B=−(dD_dark+J_B)/s,
    [□−2ab/(a+b)]D_dark=J_D−(d/s)J_B.                    (26)

There is one regular scalar wave after an actual algebraic constraint, not an additional gauge mode or a second cone inferred from a zero determinant. The constrained variable is fixed, not an arbitrary gauge function. At κ<0 the kinetic form has one negative direction; regular field redefinitions cannot remove that inertia. Neither PT nor projection rescues it. These are parameter-boundary analyses of the same parent family, not parameter changes used to compare P1 and P2.

## 8. Classification

| Comparison | Classification | Reason |
|---|---|---|
| P1 vs P2 with exact elimination, full reconstructed observables and same sources/state | **SAME_REPRESENTATION_CLASS** | Eq.(23), constant regular canonical transition and matching geometric trace response |
| Each map's own normalized retained source/detector | **DIFFERENT_PREPARATIONS_ONLY** | v1≠v2; different spectral weights compare different operations |
| P1/P2 promoted to two autonomous sectors in the generic parent | **NOT_WELL_DEFINED** as that claim | Both off-blocks are nonzero; strict deletion is not invariant evolution |
| Different strict one-pole answers presented as exact same-protocol physics | **NOT_WELL_DEFINED** | Nonzero complementary source/history/contact information was dropped |
| Distinct admissible physical sectors/universes generated by these two maps | **DIFFERENT_ADMISSIBLE_PHYSICAL_SECTORS not established** | No invariant difference survives the fixed-parent, matched-data comparison |

This is a proved limited fixed-parent no-go for the proposed projection-based multiplicity in this model, accompanied by an explicit nonzero geometric response and nontrivial Schur calculation. It is not a claim that all conceivable interacting theories lack genuine superselection sectors.
