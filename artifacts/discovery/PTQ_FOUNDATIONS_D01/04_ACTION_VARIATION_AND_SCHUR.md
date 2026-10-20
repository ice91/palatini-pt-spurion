# Actual scalar action variation, sole revision, and Schur reduction

## First derivation from the NEW independent-dual action

With signature (−,+,+,+), NEW Candidate 2 starts with

    S_0=∫d^4x[−∂μtildeΨ^T∂^μΨ−tildeΨ^T MΨ],
    M=[[a,ig],[ig,b]],  a,b,g∈R.

Vary Ψ and tildeΨ independently, integrating actual compact-support boundary terms. The Euler–Lagrange equations are

    (□I−M)Ψ=0,
    (□I−M^T)tildeΨ=0.

`derive_models.py` differentiates this Lagrangian with respect to independent field/derivative functions in a t,x sector; the two omitted spatial derivatives add identically by the declared Lorentz contraction. It does not import an equation provider. The four-dimensional formula follows directly from the same variation.

For a Fourier mode exp(−iωt+ik·x), z=ω²−|k|²,

    D(z)=[[z−a,−ig],[−ig,z−b]],
    det D=(z−a)(z−b)+g²,
    λ±=(a+b±sqrt((b−a)²−4g²))/2.

This action is a variational carrier. It is not a real positive physical action under naive tildeΨ=barΨ when g≠0; M†≠M. No positive norm is inferred from PT alone.

## Sole controlled revision R1: solve for the physical pairing

Ask, as an independent physical consistency criterion, whether a constant positive Hermitian internal pairing exists such that

    M†G=GM.

The criterion is an assumption about what a positive physical realization should mean, not an assumed coefficient bound. Solve the simplest witness ansatz G=I+rσ₂, σ₂=[[0,−i],[i,0]]. The off-diagonal equation is

    (b−a)r−2g=0,
    Δ=b−a>0,   r=2g/Δ,
    G=[[1,−ir],[ir,1]],  eig(G)=1±r.

Thus **G>0 iff Δ²>4g²** for this witness. The same inequality is necessary for any positive metric at nonzero g: a positive metric would make G^(1/2) M G^(−1/2) Hermitian, so M must be real-spectral and diagonalizable. The nonzero-g discriminant-zero point is defective; the broken region has complex eigenvalues. The harmless a=b,g=0 scalar matrix is a separate diagonalizable degeneracy, not an exceptional point.

The general Hermitian metric is not unique. Writing G=[[u,x+iy],[x−iy,v]] with g≠0 yields x=0 and (a−b)y=g(u+v), with additional positivity inequalities. Choosing u=v=1 is a concrete normalized witness, not a PT theorem selecting a unique inner product. This sprint adopts that witness as part of its NEW physical package, with no arbitrary function.

Now set tildeΨ^T=Ψ†G and use the **NEW revised physical action**

    S_G=∫d^4x[−∂μΨ†G∂^μΨ−Ψ†GMΨ−Ψ†Gj−j†GΨ].

It is real because G†=G and (GM)†=GM. Variation in Ψ† gives G(□I−M)Ψ=Gj and hence DΨ=j on the regular G domain. The conjugate equation follows from the real action. The metric satisfies ΘG*Θ=G, so U is compatible with this pairing and action on the fixed PT-isometric background. The equation is unchanged, but the action pairing and physical source/detector interpretation are now specified. This is the one controlled revision; it does not prove that the original standard pairing was already acceptable.

## Quadratic fluctuations, Hessians and constraints

The zero-field configuration is stationary and the theory is already quadratic, so its fluctuation action equals S_G. Write Ψ=(u+iv,w+iy)^T/√2 and q=(u,v,w,y)^T. Its kinetic density is (1/2)qdot^T K_R qdot−(1/2)∑i(∂i q)^T K_R(∂i q), with

    K_R=[[1,0,0,r], [0,1,−r,0],
         [0,−r,1,0], [r,0,0,1]].

Its eigenvalues are 1+r,1−r, each twice; det K_R=(1−r²)². The actual mass Hessian V_R is obtained by differentiating Ψ†GMΨ and equals K_R M_R, where

    M_R=[[a,0,0,−g], [0,a,g,0],
         [0,−g,b,0], [g,0,0,b]].

V_R is symmetric. Exact symbolic differentiation, including the opposite spatial Hessian, verifies these statements. At |r|<1 there are no velocity constraints in this scalar sector: momenta π=K_R qdot can be inverted. There are **four real propagating scalar components**, equivalently two complex scalars. The complementary scalar is dynamical, not an auxiliary connection variable. No gravitational degree-of-freedom count follows.

## Equation-level Schur complement

For arbitrary sources j=(j1,j2), the complementary equation gives

    (□−b)χ=igφ+j2,
    χ=ig R_b φ+R_b j2+χ_hom,
    R_b=(□−b)^(−1) with the declared state/domain.

Substitute into the first equation:

    [□−a+g²R_b]φ=j1+ig R_b j2+igχ_hom.

Thus from the **derived** blocks A=□−a, B=C=−ig, F=□−b,

    D_eff=A−BF^(−1)C=□−a+g²(□−b)^(−1),
    D_strict=PDP|P=□−a.

The correction is nonzero at g≠0. It is a mass-dimension-two operator; g has dimension two. In z notation D_eff=z−a+g²/(z−b), and (z−b)D_eff=det D. The literal strict projection deletes g at equation level and does not reproduce this rational dynamics.

For a retarded response take past-compact sources with zero incoming χ_hom. The inverse is a specified causal Green operator, not a reciprocal at a pole. Alternatively compare resolvents at common non-real frequency. A different initial complementary state supplies the χ_hom source above; exact elimination does not erase degrees of freedom or generate a universal state-independent theory. At z=b the scalar Schur chart fails pointwise although det D=g²≠0: the full operator need not be singular there.

## Physical-action Hessian and its different Schur complement

After R1 the **variational** Fourier Hessian is H=GD, not bare D. Using rΔ=2g,

    H11=z−a−rg,
    H22=z−b+rg,
    H12=−i[g+r(z−b)],
    H21=i[g+r(z−b)].

These are Hermitian for real z. Its variational Schur complement is

    H_eff=H11−H12 H22^(−1)H21
         =(1−r²)[(z−a)(z−b)+g²]/(z−b+rg).

It has the same two physical zero locations as the full system wherever the block inverse is valid, but differs from the equation-level D_eff. Correspondingly, **restricting the revised action to χ=0** gives H_strict=H11, not D_strict. This distinction prevents treating an unpaired equation restriction as the physical variational truncation.

For the NEW physical source j=e1 J, e1=(1,0)^T, the variational source is v=Ge1, not merely e1; the detector is e1†GΨ. Let

    v_eff=v1−H12 H22^(−1)v2,
    v_eff,L=v1*−v2* H22^(−1)H21.

The response calculated after stationary elimination is

    R_phys=v_eff,L H_eff^(−1)v_eff+v2*H22^(−1)v2
          =e1†G D^(−1)e1.

The second term is the necessary source contact contribution. Direct symbolic variation/inversion and the independent component solve agree. Neglecting source dressing would compare different physical preparations. Artificial H22 chart poles cancel in the full expression; similarly R_phys(z=b)=2/Δ is finite for g≠0.

The revised strict action with matched remaining source/detector has one pole R_phys,strict=1/(z−a−rg). The complete response has two poles and cannot equal this truncation at g≠0. Its first high-frequency mass moment is a+rg, consistent with compression to the normalized e1 line; equality of one moment does not equate the full responses.

## PT compatibility and the inverse prescription

At the formal operator/real-frequency level UDU^(−1)=D, and the two imaginary off-block factors give a PT-even Schur insertion, conditional on transporting the complementary inverse. The grading Θ alone does not commute with D at g≠0. A retarded inverse is not invariant under spacetime time reversal:

    U R_ret U^(−1)=R_adv,
    U D_eff,ret U^(−1)=D_eff,adv.

The elementary oscillator kernel θ(t)sin(Ωt)/Ω maps at t→−t to the advanced kernel −θ(−t)sin(Ωt)/Ω; this is independently checked. Thus the physically causal retarded construction is PT-covariant as a pair of prescriptions, not a single PT-fixed inverse. Formal PT-evenness of an insertion must not be confused with a proven invariance of a chosen causal state.

## Nonredundancy and low-energy caution

The exact correction is not a total derivative: its variational equation/response has a nonzero rational contribution. A local invertible nonsingular field change that also transports sources cannot remove one of two physical response poles. A nonlocal inverse-dependent transformation or deletion of a complementary state can hide that pole only by changing the comparison. The **full** two-field model is nevertheless equivalent to an ordinary Hermitian two-field control (report 06).

When |z|≪b, b≠0,

    D_eff=−a−g²/b+(1−g²/b²)z−(g²/b³)z²+O(z³/b⁴).

At a fixed low derivative order, mass/normalization shifts and terms proportional to the leading equation can be absorbed by allowed perturbative redefinitions with matched sources. No independent unique NLO operator is claimed. Treating the z² truncation as an exact UV theory can create a spurious extra ghost-like root outside |z|≪b; report 05 gives a controlled counterexample. The exact Schur operator, not an uncontrolled truncation, carries the pole comparison.
