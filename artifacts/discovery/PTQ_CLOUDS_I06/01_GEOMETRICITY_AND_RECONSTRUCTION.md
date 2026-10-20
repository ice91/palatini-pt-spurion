# I06 — one operationally geometric relaxation of simplicity

**Finding:** within one explicitly defined real non-chiral BF family, nineteen simplicity conditions suffice to reconstruct a Lorentzian metric plus one scalar. Full simplicity sets that scalar to zero. A metric electromagnetic propagation law can be constructed directly from B and is insensitive to this scalar. These are conditional geometric results, not a derivation of the remaining nineteen constraints or of a universal matter law.

## 1. Fixed domain and one deformation

Retain I05's oriented local four-manifold, real SO⁺(1,3) bundle, η_IJ=diag(−1,1,1,1), ε_0123=+1 and real fields

\[
 A^{IJ}\in\Omega^1(\mathrm{ad}P),\quad B^{IJ}\in\Omega^2(\mathrm{ad}P),
 \quad \phi_{IJKL}=\phi_{KLIJ},\quad\nu\in\Omega^4(M).
\]

A and B are antisymmetric in each internal pair; φ has 21 components. The connection is metric compatible on the internal bundle. The internal star satisfies star²=−I. It is not initially a spacetime Hodge star. All fields, coframes reconstructed below, and the deformation are real; no chiral complex-sector reality conditions are being presumed.

Use compact variations, or fixed connection variation at the boundary as in I05. Calculations concern local bulk modes, not boundary charges or global coframe topology. The common linearization background is exactly I05's Ā=0, B̄=star(dx∧dx), φ̄=0, ν̄=−vol, with reconstructed Minkowski metric and zero cosmological parameter. The physical branch has positive coframe orientation, κ>0, future time orientation and ν<0. These are named domain choices, not conclusions of operational consistency.

Define the invariant identity tensor and scalar multiplier trace

\[
 q_{IJKL}=\eta_{I[K}\eta_{L]J},\quad
 \tau=q^{IJKL}\phi_{IJKL},\qquad q\cdot q=6,\quad q\cdot\epsilon=0.
                                                               \tag{1}
\]

**NEW CANDIDATE DEFINITION:** soften only this lowest-rank invariant multiplier direction:

\[
 \boxed{S_\lambda=\frac1{2\kappa}\int
 [B^{IJ}\wedge F_{IJ}-\tfrac12\phi_{IJKL}B^{IJ}\wedge B^{KL}
 -\tfrac\nu2\epsilon^{IJKL}\phi_{IJKL}-\tfrac{\lambda\nu}{2}\tau^2].}
                                                               \tag{2}
\]

λ is the sole new constant, of dimension length²; κ remains I05's free normalization. λ=0 is precisely the hard action. No matter spectrum was targeted to choose a Pauli–Fierz tuning. The independent motivation is the smallest nontrivial invariant relaxation—one scalar channel while leaving the other multiplier directions linear. Its spectrum/sign must be calculated, not presumed. This is **not** an arbitrary full-rank penalty on all twenty multipliers. The multiplier Hessian has rank one, so a generic full-rank non-chiral degree count cannot be applied. No free constitutive function has been introduced.

## 2. All field variations and exact algebraic content

Variations of all independent fields give

\[
 D_AB=0,\qquad F_{IJ}=\phi_{IJKL}B^{KL},                     \tag{3}
\]
\[
 B^{IJ}\wedge B^{KL}+\nu\epsilon^{IJKL}
                  +2\lambda\nu\tau q^{IJKL}=0,\qquad
 \epsilon\cdot\phi+\lambda\tau^2=0.                         \tag{4}
\]

The new term has no B or A dependence with ν independent. Contracting (4) with epsilon and q yields

\[
 \nu=V_B:=\tfrac1{24}\epsilon_{IJKL}B^{IJ}\wedge B^{KL},
 \quad Q_B:=q_{IJKL}B^{IJ}\wedge B^{KL},\quad
 Q_B+12\lambda V_B\tau=0.                                   \tag{5}
\]

For λ≠0 and V_B≠0, one equation determines τ; the nineteen pair-symmetric components orthogonal to both epsilon and q still impose hard constraints. The epsilon component of φ is fixed by the last equation of (4), rather than silently kept zero. Hard λ=0 instead imposes Q_B=0 as its twentieth simplicity condition.

Use ordered pair basis (01,02,03,12,13,23), wedge matrix W, D=diag(−1,−1,−1,1,1,1), internal star matrix S=DW. Write G=bWbᵀ for the Gram matrix of B. In coefficients of a chosen four-volume,

\[
 v=\tfrac16\sum_{ab}W_{ab}G_{ab},\quad Q=2\operatorname{tr}(DG),
 \qquad C_{19}(B):=G-vW-\frac Q{12}D=0.                      \tag{6}
\]

For invertible b the derivative δb↦δG is onto the 21-dimensional symmetric matrices: δb=(1/2)δG(bᵀ)⁻¹W⁻¹ supplies any δG. Removing the two independent invariant directions W,D therefore gives **rank 19 throughout this regular region**, not just at the background. Hard projection onto the complement of W gives rank 20 instead. This is a pointwise algebraic rank, not a physical degree count.

## 3. Reconstructed metric plus scalar, without an extra metric field

An exact solution of (6) near the selected gravitational branch is

\[
 \boxed{B^{IJ}=\star\Sigma^{IJ}+\beta\Sigma^{IJ},\qquad
 \Sigma^{IJ}=e^I\wedge e^J,\quad\det e\ne0.}                 \tag{7}
\]

Direct exterior multiplication gives

\[
 G=(\beta^2-1)W+2\beta D,\quad
 V_B=(\beta^2-1)\operatorname{vol}_e,\quad
 Q_B=24\beta\operatorname{vol}_e,\quad
 \det(S+\beta I)=(1+\beta^2)^3.                              \tag{8}
\]

Thus β is not a new matter field or independent metric; it parametrizes the one released component of B. It can be extracted invariantly on the ν<0, positive-orientation chart. Put r=Q_B/(24V_B); then

\[
 r=\frac\beta{\beta^2-1},\qquad
 \beta=-\frac{2r}{1+\sqrt{1+4r^2}},\qquad |\beta|<1.           \tag{9}
\]

The reciprocal solution belongs to a different sheet and is not silently merged. The derivative dr/dβ=−(1+β²)/(1−β²)² never vanishes here. From β and B, recover Σ=(S+βI)⁻¹B and hence a local real coframe using I05's simple-form reconstruction, modulo its discrete/frame choices. The metric is **derived from this constrained B**, not added as an unrelated gravitational variable. Its signature comes from the already supplied internal Lorentz metric.

The coframe-plus-scalar map has rank 17: δB=0 first fixes δβ=0 by (9), then δΣ=0 by invertibility of S+βI, then δe=0 by the exterior-square tangent map. This matches 36−19=17. The implicit-function theorem therefore makes (7) a complete local parametrization of the regular constraint surface near the chosen branch, not a two-mode truncation. Explicit symbolic ranks at β=0,1/2 and independent non-diagonal coframe tests check the conventions; the surjectivity argument supplies the generic rank theorem.

Equation (5) now fixes

\[
 \tau=\frac{2\beta}{\lambda(1-\beta^2)},\quad
 \frac{d\tau}{d\beta}=\frac{2(1+\beta^2)}{\lambda(1-\beta^2)^2}.
                                                               \tag{10}
\]

For finite nonzero λ this algebraic inversion is regular on |β|<1, including β=0. A hidden extra multiplier mode is not inferred from a vanishing background τ. Hard simplicity in this chart is exactly β=0.

## 4. Operational matter map constructed directly from B

For invertible B define the typed bundle map

\[
 \mathcal B:\Lambda^2E\longrightarrow\Lambda^2T^*M,\qquad
 v^{IJ}\mapsto\tfrac12 B_{IJ}v^{IJ},\qquad
 J_B=\mathcal B\,\star\,\mathcal B^{-1}.                    \tag{11}
\]

The pair-coordinate matrix of mathcal B is bᵀD; the factors and lowered indices matter. This map is real, internal-Lorentz invariant after conjugation, and diffeomorphism covariant. It satisfies J_B²=−I wherever invertible. That identity alone for arbitrary B does **not** prove that J_B is a metric Hodge operator, positive pairing or hyperbolic propagation.

For (7), S+βI commutes with S, so conjugation cancels its β dependence. At e=dx the explicit result is J_B=−S; this is precisely the spacetime Lorentz Hodge star with the chosen orientation. Covariance gives

\[
 \boxed{J_B=*_{g(e)}\text{ on two-forms, independently of }\beta.} \tag{12}
\]

Its wedge adjoint satisfies J_BᵀW=WJ_B. A separate rational calculation verifies (12) against the direct metric Hodge formula for a non-diagonal coframe and three different β values. Since a two-form Hodge operator in four dimensions is conformally invariant, this operational map determines a conformal metric structure; it does not by itself determine lengths, β, or the multiplier potential.

**ASSUMED PROBE COUPLING, with derived consequences:** choose one U(1) Maxwell probe a, f=da,

\[
 S_{EM}[B,a]=-\tfrac12\int f\wedge J_Bf.                     \tag{13}
\]

No fitted constitutive function is used; (11) is the exact B-dependent constitutive operator chosen by this new model. On (7), (13) becomes the ordinary Maxwell action on g(e), and its characteristic cone and positive energy are derived in report 02. The coupling law (13) itself is an independent physical assumption. It is not an inference from observers demanding a cone.

For a clock/scale comparison only, one massive real scalar probe may be minimally coupled to the same reconstructed g(e). This is explicitly an **ASSUMED shared metric prescription**; its matching cone does not prove universal coupling. No second independently varied gravitational metric or arbitrary β-dependent matter function is introduced. Maxwell alone cannot measure a conformal length scale or enforce β=0.

## 5. Other branches and rank-changing boundaries

The hard theory retains I05's B=±Σ (topological), B=±starΣ (gravitational), and degenerate branches. Softening does not authorize deleting them. An alternate chart of the **same** nineteen-constraint family is B=Σ+γstarΣ. It has V=(1−γ²)vol and Q=24γvol. At γ=0, (5) enforces τ=0; F=φΣ with φ=Riemann/2 then additionally requires scalar R=0. The entire arbitrary-metric hard topological family therefore does not remain unchanged. Scalar-flat torsion-free examples, including Minkowski, remain exact. Their TT kinetic coefficient vanishes at γ=0; no healthy two-graviton interpretation is assigned there. Report 02 displays its reduced action and singular linear test; a nonlinear count on that singular stratum is not guessed.

The sign-reversed gravitational chart has reversed gravitational/scalar kinetic signs relative to the positive matter normalization, while the multiplier-generated potential depends quadratically on B and does not simply reverse with B. It is not admitted into the healthy result by changing names. Orientation and sign are held fixed on the primary branch.

For finite multipliers, V_B=ν=0 in (4) forces all B∧B=0. The B span is isotropic and has rank≤3; no inverse mathcal B exists. B=0 and A=0,φ=0,B^{01}=dx⁰∧dx¹ remain exact degenerate solutions. Our inverse-defined probe law is not defined there. This is a necessary invertibility condition **for this probe law**, not a proof that nature forbids every alternative degenerate-phase coupling.

At β=±1 formula (8) alone would give invertible B and V=0, but Q≠0 violates (5) for finite τ. The pole in (10) is a boundary of this finite-field model chart, not an extra mode. Other orientation sheets/global sector classifications are not merged into the local positive branch.

## 6. Evidence and source discipline

`derive_geometry.py` executes 21 exact reconstruction/connection checks; `check_independent.py` independently uses rational exterior-action arithmetic. No original I05 result is counted as a new experiment. The I05 hard result is retained as the control. This one-singlet deformation has a dynamically varying Holst-like component, but it is derived here from (2), not inserted as an EH starting action. The kinetic benchmark is compared to Taveras–Yunes [0807.2652](https://arxiv.org/pdf/0807.2652), Eq.(49); our rational potential comes specifically from (2). Neither that paper nor a chiral result proves the present model's microscopic origin.
