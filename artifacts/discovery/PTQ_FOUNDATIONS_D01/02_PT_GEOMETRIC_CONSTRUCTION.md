# Natural geometric PT: construction and limited no-go

Everything in this section concerns **NEW Candidate 1**, not a repaired reading of Symmetry. Conclusions are NEW CONDITIONAL MATHEMATICAL RESULTS on the declared domain.

## Explicit lift and Cartan compatibility

Choose the real trivial Lorentz bundle on R^4 and coordinate involution θ:x→−x. For a complexified coefficient-valued p-form α define

    Uα=θ*bar(α).

On the actual real e,ω sector conjugation is trivial. For complex coefficients U is real-linear and anti-complex-linear; its action preserves product order. Internal Lorentz indices are unchanged in this fixed bundle trivialization. Since θ²=I and pullback commutes with complex conjugation,

    U²α=α,
    U(dα)=d(Uα),
    U(α∧β)=Uα∧Uβ.

The last identity does not exchange factors. Because η_AB and the internal ε_ABCD are real constants, their contractions are equivariant. Applying these identities directly to the Cartan definitions gives

    U T(e,ω)=d(Ue)+(Uω)∧(Ue)=T(Ue,Uω),
    U R(ω)=d(Uω)+(Uω)∧(Uω)=R(Uω).

The connection remains so(1,3)-valued because pullback/conjugation preserves its real antisymmetric lowered Lorentz pair. No affine projective shift is invoked. The metric is induced consistently:

    g=η_AB e^A⊗e^B,   Ug=θ*g.

For a complexified non-real metric one would also transport its conjugate; this sprint uses real Lorentz metrics, avoiding an unproved complex-Hodge branch.

## Orientation, Hodge and boundaries

Four coordinate sign reversals give det dθ=+1, hence spacetime and four-coframe orientations are preserved. The pulled-back flat e^0=−dt reverses time orientation relative to the fixed ∂t, and the spatial coframe orientation also reverses. The configuration domain includes both time orientations; if one instead demanded future coframes at fixed background time orientation, U would map to a paired domain rather than be an endomorphism of that smaller domain. We do not conceal that distinction by an unstated internal Lorentz transformation.

For real oriented metrics, Hodge naturality is

    θ*(*_g α)=*_{θ*g}(θ*α),
    U(*_g α)=*_{Ug}(Uα).

This is a map on geometry-and-form pairs. Only when θ*g=g may it be written [U,*_g]=0 at fixed background. Flat Minkowski is such a background. The independent exterior-algebra implementation checks involution, exterior derivative, wedge, Cartan torsion/curvature, and Minkowski Hodge on nonconstant sample forms/coframes/compatible connections; the abstract pullback proof covers general smooth forms on the declared domain.

The volume form transforms naturally: Uvol_g=θ*vol_g=vol_{θ*g}. Integrals obey ∫_{θD}θ*L=∫_D L for real top forms and an orientation-preserving diffeomorphism; complex coefficient integrals are conjugated. For action symmetry at a fixed boundary, the domain/data must also be invariant or transported. Compactly supported variations suffice for the variational calculation. A retarded state is not invariant under time reversal; it is transported to advanced data. No global PT isometry of arbitrary curved fixed backgrounds is assumed.

## Lemma 1 — natural-pullback covariance no-selection

**Statement.** Let L(e,ω) be any real-coefficient natural local four-form formed from real e,ω, their Cartan tensors, covariant derivatives, η, internal orientation and metric Hodge, with actual index/degree contractions. With transformed backgrounds/domains as above,

    L(Ue,Uω)=θ*L(e,ω).

Consequently every real linear combination of such terms has the same transformed-action property. Requiring this natural PT covariance imposes **no new real numerical coefficient equation** beyond ordinary diffeomorphism/bundle covariance on this class.

**Proof.** Each generator and each allowed operation is natural and real. Induction over the tensor/wedge/derivative/contraction expression gives the displayed identity. The orientation-preserving change of variables gives the action identity. Coefficient values never enter the argument. QED.

This is deliberately a LIMITED no-go: fixed external non-invariant data, anti-linear actions on nontrivial internal representations, genuinely complex coefficients, orientation-reversing transformations, nonlocal operators or unmatched boundaries fall outside its hypotheses. A component being odd is not excluded: the integrated top-form functional is invariant after transport. This is not a claim that every field density is pointwise θ-even on every background.

## Actual Einstein–Cartan variation

NEW action, with no assumed torsion specialization:

    S_EC=(M_P²/4)∫ε_ABCD e^A∧e^B∧R^{CD}(ω).

Use δR^{CD}=D_ω δω^{CD}. Integration by parts, D_ωε=0 and D_ωe^A=T^A give the connection equation, up to a nonzero common normalization,

    ε_ABCD e^C∧T^D=0.

The e variation independently gives

    ε_ABCD e^B∧R^{CD}=0.

At a nondegenerate coframe the first is an invertible 24-component algebraic system for torsion: in a coframe basis its six antisymmetric AB values and four 3-form components form a 24×24 integer matrix. `derive_EC_connection.py` constructs it directly from ε and wedge signs and obtains **rank 24, determinant 16**. A change of nondegenerate frame cannot change invertibility. Therefore T^A=0 follows; pure trace was never imposed. The second equation reduces to vacuum Einstein equations on that solution. e^A=dx^A,ω=0 is an explicit consistent branch.

These are standard Einstein–Cartan equations newly reconstructed/tested here, not a claim of a new gravity theory. They do not establish the old Symmetry C1 coefficient formula. H0 survives in the stated natural-real class, so the second and final candidate introduces a separate internal anti-linear action on a probe carrier.
