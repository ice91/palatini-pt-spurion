# I07 — a hard-simple microscopic parent and a covariant block map

**Decision:** the declared linear block map does generate nonsimple effective B, but does **not** close on I06's C19 surface. The failure occurs for a smooth exact flat solution of the hard microscopic equations, with all microscopic and coarse bivector maps invertible. No effective scalar spectrum is inferred. This is a newly executed conditional mathematical result, not a microscopic origin of gravity.

## 1. Fixed microscopic dynamics and assumption budget

Use the same real, non-chiral Lorentzian field types as I05: an oriented local four-manifold M, an SO⁺(1,3) bundle with internal η=diag(−1,1,1,1), ε₀₁₂₃=+1, real metric-compatible connection A, real adjoint two-form B, pair-symmetric multiplier φ and real four-form ν. Internal star has star²=−I. No extra metric, matter field, scalar potential or penalty is supplied.

**MICROSCOPIC MODEL CHOICE:** retain the hard Plebanski action, with zero cosmological parameter,

\[
 S_{micro}=\frac1{2\kappa}\int_M
 [B^{IJ}\wedge F_{IJ}-\tfrac12\phi_{IJKL}B^{IJ}\wedge B^{KL}
 -\tfrac\nu2\epsilon^{IJKL}\phi_{IJKL}],\qquad \kappa>0.       \tag{1}
\]

Here antisymmetric pairs are summed over all ordered internal indices. Use smooth fields, compact variations, or fixed boundary connection variation. The hard equations are

\[
 D_AB=0,\quad F_{IJ}=\phi_{IJKL}B^{KL},\quad
 B^{IJ}\wedge B^{KL}+\nu\epsilon^{IJKL}=0,\quad
 \epsilon^{IJKL}\phi_{IJKL}=0.                              \tag{2}
\]

These are the established I05 action and equations, not new microscopic laws derived in I07: `artifacts/discovery/PTQ_CLOUDS_I05/01_GEOMETRIC_PARENT_AND_VARIATION.md`, Eqs.(4)–(9). Hard simplicity is explicitly **assumed**. Work on its positive, real, nondegenerate gravitational branch B=star(e∧e), det e>0, with the selected future orientation; other branches are not claimed excluded by physics. No quantum measure or statistical ensemble is constructed. An explicit classical action is the microscopic dynamics used for this first closure test. The counterexample below solves its equations; it is not an arbitrary inconsistent junction of two solutions.

| Independently supplied input | Motivation and role | What it does not establish |
|---|---|---|
| Real Lorentz bundle and hard action (1) | Fixed I05 geometric parent and reality/gauge domain | Origin or necessity of microscopic simplicity |
| Positive gravitational branch, κ>0, zero cosmological parameter | Controlled regular comparison, admitting an exact flat solution | Selection of that branch or a finite Newton constant |
| Two reference-cell embeddings, comparison paths and normalized weights | Finite-resolution comparison of local two-forms | A relational apparatus derived from gravity, a UV completion or a preferred vacuum distribution |
| Parallel transport from A | Required to compare distinct internal fibers covariantly | Preservation of nonlinear algebraic constraints under averaging |

The cell size ℓ is a supplied observation scale, not a microscopic mass or an inserted scalar potential. No independent physical UV scale, λ, nineteen penalties, or constitutive function is introduced.

## 2. One typed coarse-graining map

**NEW CANDIDATE DEFINITION:** let Q be an oriented reference cell, f₁,f₂:Q→M two smooth embeddings of equal reference-coordinate size, f₀=f₁ the root embedding, and γᵢ(ξ) specified paths from fᵢ(ξ) to f₀(ξ). Let Hᵢ(ξ) be connection parallel transport along these paths, mapping the internal vector fiber at fᵢ(ξ) to the root fiber. H₁=I for the constant path. Fix w=1/2 for the main calculation; varying w below tests the same normalized linear map for robustness.

\[
 \boxed{\mathcal C_{w,f,\gamma}(A,B)=\bar B
   =(1-w)\operatorname{Ad}_{H_1}f_1^*B
      +w\operatorname{Ad}_{H_2}f_2^*B.}                    \tag{3}
\]

Its domain is smooth microscopic connection/two-form data **together with** the declared comparison protocol. Its codomain is Ω²(Q,ad(f₀*P)); it does not have the hard-simple or C19 submanifold as an assumed codomain. It is real-linear in B at fixed A and protocol, generally nonlinear in A. It is not an idempotent endomorphism: its domain and codomain differ.

Use A′=gAg⁻¹−dg g⁻¹, B′=Ad_g B. Parallel transport transforms as Hᵢ′=g(f₀)Hᵢg(fᵢ)⁻¹. Thus

\[
 \bar B' =\operatorname{Ad}_{g(f_0)}\bar B.                \tag{4}
\]

No internal tensors in unrelated frames have been added. Both spacetime form slots are compared by the **specified pullbacks** Dfᵢ, rather than treating coordinate components at different points as the same tensor without a map. Under an active diffeomorphism represented by ψ* on fields, transport the entire protocol by fᵢ→ψ⁻¹∘fᵢ and the corresponding paths; (3) is then covariant. Keeping the embeddings fixed while changing the fields changes the supplied sampling protocol relative to the geometry. We do not claim a diffeomorphism-invariant bulk observable of gravity alone without boundary/reference apparatus.

Accessible data in this calculation are the path-dressed coarse two-form relative to a boundary/root frame and internal-frame-invariant contractions of its wedge Gram matrix, including v,Q and the residual invariant in §5. These are coarse geometric observables **conditional on the supplied protocol**, not a quantum observer algebra or an independently derived matter coupling. No physical time averaging is required by the example.

## 3. Connected, smooth and exactly on-shell microscopic witness

This construction avoids discontinuous coframes and unverified interface equations. Use coordinates (t,x,y,z). Choose a smooth step h(u), equal to 0 for u≤0 and 1 for u≥1, with 0≤h≤1. One explicit choice is h(u)=ρ(u)/[ρ(u)+ρ(1−u)], where ρ(u)=0 for u≤0 and exp(−1/u) otherwise. Set

\[
 f(x)=1+h(x/\ell+1/2),\quad g(x)=1+2h(x/\ell+1/2),
 \quad G'(x)=g(x),
\]
\[
 e^0=d[f(x)t]=f(x)dt+t f'(x)dx,\quad e^1=dG(x)=g(x)dx,
 \quad e^2=dy,\ e^3=dz.                                  \tag{5}
\]

All four e's are closed. Their determinant is f g>0, and the metric η_IJ e^I e^J is the pullback of Minkowski space by X⁰=f(x)t, X¹=G(x), X²=y, X³=z. Thus it is exactly flat, not a curvature excitation. Choose a bounded slab |t|<T with T‖f′‖∞<1; its t-slices are spacelike since g²−t²f′²>0. Coordinates and the time orientation are regular.

Take

\[
 A=0,\quad \phi=0,\quad B=\star(e\wedge e),\quad
 \nu=-f g\,dt\wedge dx\wedge dy\wedge dz.                 \tag{6}
\]

Equations (2) hold everywhere: de=0 gives D_AB=dB=0; F=0=φB; the two multiplier equations follow from hard simplicity and φ=0. All interface derivatives are smooth and included. This exact solution is used to test the deterministic map, not claimed to dominate an unspecified path integral.

Let Q have |ξˣ|,|ξʸ|,|ξᶻ|<ℓ/4 and |ξᵗ|<T, with

\[
 f_1(\xi)=(\xi^t,\xi^x-2\ell,\xi^y,\xi^z),\quad
 f_2(\xi)=(\xi^t,\xi^x+2\ell,\xi^y,\xi^z).               \tag{7}
\]

Use straight comparison paths at fixed t,y,z. Because A=0 all Hᵢ=I. The two pullback coframes are exactly e₁=diag(1,1,1,1) and e₂=diag(2,3,1,1). Their determinant signs, time orientation and branch sign agree. Points with the same ξᵗ are compared at the same coordinate time. The second cell samples larger physical clock/length units relative to the supplied reference embeddings. Consequently a nonsimple coarse B in this witness cannot automatically be interpreted as a new physical excitation: the underlying connected geometry is still flat.

## 4. Exact closure calculation

Use the ordered pair basis (01,02,03,12,13,23), relative to the reference four-volume. Define

\[
 W_{(IJ)(KL)}=\epsilon_{IJKL},\quad
 D=\operatorname{diag}(-1,-1,-1,1,1,1),\quad S=DW,
 \quad S^2=-I.
\]

For a coframe matrix e, U(e)=Λ²e. The coefficient matrix of hard gravitational B is b=S U(e). For any b define

\[
 \mathcal G=bWb^T,\quad v=\tfrac16\operatorname{tr}(W\mathcal G),
 \quad Q=2\operatorname{tr}(D\mathcal G),\quad
 C_{19}=\mathcal G-vW-(Q/12)D.                            \tag{8}
\]

This is exactly the I06 diagnostic (`01_GEOMETRICITY_AND_RECONSTRUCTION.md`, Eq.(6)), not a new constraint imposed here. Microscopic b₁=S and b₂=S diag(6,2,2,3,3,1) each have Q=0 and \(\mathcal G_i=-\det(e_i)W\).

Applying (3) gives

\[
 \bar b=S\operatorname{diag}(1+5w,1+w,1+w,1+2w,1+2w,1).
                                                               \tag{9}
\]

Its three complementary-pair products are

\[
 t_1=1+5w,\quad t_2=t_3=(1+w)(1+2w),\quad
 \delta=t_1-t_2=2w(1-w).                                 \tag{10}
\]

Direct exterior multiplication yields Q=0, v=−(t₁+2t₂)/3, and the only independent nonzero C19 entries are

\[
 (C_{19})_{01,23}=-2\delta/3,\quad
 (C_{19})_{02,13}=-\delta/3,\quad
 (C_{19})_{03,12}=+\delta/3,                              \tag{11}
\]

with symmetric partners. These entries comprise a trace-free complementary-pair distortion; they are among the nineteen directions that I06 keeps constrained, not its released q-singlet. For every genuine two-cell mixture 0<w<1 they are nonzero. Meanwhile

\[
 \det\bar b=(1+5w)(1+w)^2(1+2w)^2>0,\qquad v<0.           \tag{12}
\]

No degenerate coframe, bad real form, opposite branch or zero-volume boundary explains the failure.

## 5. Gauge-invariant obstruction to the single-scalar target

At w=1/2, \(\bar b=S\operatorname{diag}(7/2,3/2,3/2,2,2,1)\), with

\[
 v=-19/6,\quad Q=0,\quad\det\bar b=63/2,\quad
 \operatorname{spec}(\mathcal G W)=\{-7/2\ (2),-3\ (4)\}.
                                                               \tag{13}
\]

The Lorentz-frame-independent residual scalar

\[
 \mathcal I_{19}:=\frac{\operatorname{tr}[(C_{19}W)^2]}{v^2}
     =\frac{12}{361}\ne0                                 \tag{14}
\]

also cancels the reference four-volume normalization. Under a root frame R=Λ²Λ, C19→R C19 Rᵀ and RᵀWR=W, so the trace is invariant. This is a diagnostic, **not** a positive norm on all Lorentzian residual tensors. Nonzero (14) proves C19≠0; a vanishing value would not by itself prove all constraints. Rational nontrivial Lorentz transformations at both cells and at the root, with their transformed links, independently preserve this result.

For I06's actual family B=(star+β)Σ(e), v=(β²−1)det e and Q=24β det e. Its regular coframe determinant is nonzero. Therefore Q=0 would force β=0, hence full simplicity and C19=0. Equations (11)–(14) contradict that consequence. No change of coframe, internal Lorentz frame or root normalization turns this averaged B into the I06 family. Extracting β=0 from Q/v **without** checking C19 would be a false reconstruction.

This is a failure of closure of the specified coarse variable, not a theorem that every nonsimple B lacks an operational metric or that no alternative coarse variable can be geometric.

## 6. Scope of the result

The basic fact that sums of simple data need not be simple is not new physics. I05 already exhibited a related two-coframe algebraic sum (`03_ORIGIN_OF_GEOMETRIC_DYNAMICS.md`, §4); it is not counted as a new independent experiment here. I07 adds the typed transport/pullback map, normalized-weight obstruction throughout 0<w<1, a connected exact-solution witness, C19 versus scalar discrimination, and independent new calculations. The stronger I07 target fails before any spectrum is warranted.

**EARLY STOP: C19_CLOSURE_FAILURE.** No second microscopic mechanism, artificial nineteen-direction penalty, projected repair or speculative mass computation is launched.
