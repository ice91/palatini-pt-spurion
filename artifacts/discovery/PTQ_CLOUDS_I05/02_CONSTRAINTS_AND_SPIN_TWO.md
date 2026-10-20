# I05 — a derived linear spin-two sector, with its domain stated

**Result:** the nondegenerate, correctly oriented positive-sign gravitational branch of report 01 has exactly two local linear vacuum polarizations about Minkowski space. They have helicities ±2, positive reduced energy and a massless retarded response. The unconstrained BF control has no local bulk propagating modes. The full simplicity action also contains other sectors, to which this two-mode result does not extend.

## 1. Same background for control and candidate

Use Ā=0, ē^I=dx^I, B̄=star(dx∧dx), φ̄=0 and ν̄=−dx⁰∧dx¹∧dx²∧dx³. Equations (5) are all satisfied; the vacuum cosmological parameter was specified as zero. The same Ā,B̄ solve the BF control. Fields, Lorentz group and BF normalization remain fixed in the comparison; the change is the simplicity-multiplier term. For the propagation calculation choose the positive coframe orientation, future-directed time and σ=+ with κ>0; formulas retain σ to display the sign failure on the other regular gravitational branch.

Take local plane waves or spatial wave packets with nonzero spatial momentum. Compact variations or decaying data remove surface terms. Global zero modes, boundary charges and transitions through ν=0 are outside this local perturbative result. No condition on modes is inferred from a finite auxiliary Hessian alone.

## 2. Why unconstrained BF has no ordinary graviton

Linearizing F=0 and D_AB=0 gives

\[
 da=0,\qquad db+[a,\bar B]=0.                               \tag{14}
\]

Besides internal Lorentz transformations, pure BF has the shift symmetry δ_ηB=D_Aη, δ_ηA=0 with η∈Ω¹(ad P). Its action changes by ∫d(η∧F), because D_AF=0. Locally da=0 implies a=dα, removable by a Lorentz transformation; after this, db=0 implies b=dη, removable by the shift. Thus every local linear solution is gauge. This is stronger than calling a B component “tensor shaped.” A would-be metric reconstructed from B in the control is not invariant under this extra shift and cannot be called a physical propagating metric there.

The canonical check starts with A_i^{IJ} and its 18 B-derived momenta, phase dimension 36. There are six Gauss constraints and 18 spatial flatness constraints. The six Bianchi identities D_iε^{ijk}F_jk=0 make only 12 of the latter independent. These first-class constraints give

\[
 N_{BF}=\tfrac12[36-2(6+12)]=0.                              \tag{15}
\]

Equivalently a nonzero spatial Fourier momentum gives rank two for the curl in each of six internal directions. Shift reducibility η=D_Aχ holds on F=0. Diffeomorphisms are on-shell combinations of the BF gauge transformations; they must not be counted as four further independent first-class constraints. This count concerns local bulk BF, not its topological/boundary state space.

## 3. What simplicity changes

Linear simplicity at the regular background forces

\[
 b^{IJ}=\star(f^I\wedge\bar e^J+\bar e^I\wedge f^J).          \tag{16}
\]

Its completeness is proved by the rank-20/16 tangent calculation in report 01. A generic BF shift dη is not tangent to (16): choose locally δB^{01}=dx⁰∧dx¹, other components zero. It is dη with η^{01}=x⁰dx¹, but at B̄ its simplicity residual δC^{01,01}=−2 vol is nonzero. Smooth compact cutoffs can give the same value in a neighborhood. The generic topological shift is broken, while Lorentz and diffeomorphism symmetries remain.

Adding an algebraic constraint can therefore **increase** propagating content relative to BF: it changes which equations/gauge freedoms survive, instead of merely removing states from a fixed physical Hilbert space. Subtracting “20 more first-class constraints” from (15) would be incorrect.

The connection equation at first order is

\[
 df^I+a^I{}_J\wedge\bar e^J=0.                              \tag{17}
\]

The 24×24 invertibility check fixes a uniquely from the derivatives of f. Its six Lorentz gauge directions act on the antisymmetric part of f_{μν}=η_Iμ f^I_ν; the remaining metric perturbation is h_μν=f_μν+f_νμ (ten components). Eliminating this connection uses its equation, not a restriction to trace torsion or a choice of configured ADM coefficients.

The coframe equation becomes the actual linearized vacuum Einstein equation derived from (10):

\[
 G^{(1)}_{\mu\nu}=\frac12\left[
 \partial_\rho\partial_\mu h^\rho{}_\nu+
 \partial_\rho\partial_\nu h^\rho{}_\mu-\Box h_{\mu\nu}
 -\partial_\mu\partial_\nu h
 -\eta_{\mu\nu}(\partial_\rho\partial_\sigma h^{\rho\sigma}-\Box h)
 \right]=0.                                                 \tag{18}
\]

It satisfies ∂^μG^(1)_μν=0 and is invariant under δh_μν=∂_μξ_ν+∂_νξ_μ. These identities are explicitly checked as polynomial Fourier identities, not inferred just from an assigned mode count.

## 4. Canonical reduction before choosing a TT basis

Use the Gauss–Codazzi identity on the **derived** torsion-free action (12), modulo its actual compact-variation boundary terms. Set N=1+n, N_i=n_i, γ_ij=δ_ij+h_ij. Its quadratic kinetic expression and momentum are

\[
 K^{(1)}_{ij}=\tfrac12(\dot h_{ij}-\partial_i n_j-\partial_j n_i),\quad
 \mathcal L_{kin}^{(2)}=\frac{\sigma}{2\kappa}(K_{ij}K_{ij}-K^2),\quad
 \Pi^{ij}=\frac{\sigma}{2\kappa}(K^{ij}-\delta^{ij}K).         \tag{19}
\]

The linear spatial scalar curvature is ∂_i∂_j h_ij−∇²h_ii. Independent n,n_i variations, rather than a two-mode truncation, give

\[
 C=\partial_i\partial_j h_{ij}-\nabla^2h_{ii}=0,\qquad
 C_i=\partial_j\Pi^{ij}=0.                                  \tag{20}
\]

Lapse/shift momenta are nondynamical and excluded from the 12-dimensional spatial metric phase space; they are not counted twice. The four linear constraints are first class. Direct brackets vanish: the mixed bracket vanishes because the double-gradient/trace combination is invariant under spatial gauge transformations; the other two brackets vanish by h-only or Π-only dependence. Their Hamiltonian preservation is the linear Bianchi identity. The four gauge generators are independent at nonzero spatial momentum.

Explicitly set momentum along z and divide only nonzero factors of k. Use canonical coordinates

\[
 q=(h_{xx},h_{yy},h_{zz},h_{xy},h_{xz},h_{yz}),\quad
 p=(\Pi^{xx},\Pi^{yy},\Pi^{zz},2\Pi^{xy},2\Pi^{xz},2\Pi^{yz}).
                                                               \tag{21}
\]

The constraint rows are (q_0+q_1,p_4,p_5,p_2); they have rank four. With the standard symplectic matrix Ω their brackets JΩJᵀ vanish and gauge matrix ΩJᵀ has rank four. Hence

\[
 \dim\Gamma_{physical}=12-4-4=4,\qquad N_{configuration}=2.   \tag{22}
\]

Spatial gauge removes h_xz,h_yz,h_zz, while the Hamiltonian gauge removes the conjugate transverse trace momentum; C removes h_xx+h_yy. The remaining canonical pairs are h_+=h_xx=−h_yy, p_+=Πxx−Πyy, and h_×=h_xy, p_×=2Πxy. The conformal negative direction of the unreduced DeWitt kinetic form is not an extra physical ghost. This is an exact linear canonical quotient, not an imposed removal of scalar/vector waves.

## 5. Independent full-symbol test of completeness

Before choosing any spatial gauge, form the 10×10 Fourier operator from (18). For q_μ=(−ω,0,0,k), its generic off-shell rank is six. On a nonzero null q, its rank is four, so its solution kernel has dimension six. The four vectors q_μξ_ν+q_νξ_μ have rank four and lie in this kernel. Adding the plus/cross tensors gives rank six and therefore exhausts the full kernel. No third physical null-wave polarization is left over.

An independent pure-rational implementation repeats this calculation for the oblique null vector q_μ=(−5,3,4,0). It constructs transverse vectors u=(−4/5,3/5,0),v=(0,0,1), and the tensors u⊗u−v⊗v and u⊗v+v⊗u. Both solve the full equations; together with gauge vectors they span the six-dimensional kernel. Their spatial pairings are 2,2,0. This independently verifies that the axis choice in (21) has not hidden a polarization.

## 6. Kinetic sign, helicity and actual physical response

Now insert the **complete reduced** tensor basis

\[
 h_{ij}^{TT}=\begin{pmatrix}h_+&h_\times&0\\h_\times&-h_+&0\\0&0&0\end{pmatrix}.
                                                               \tag{23}
\]

Computing the linear Christoffels in the ΓΓ representative of √−gR, with compact-source integration by parts, yields

\[
 [\sqrt{-g}R]^{(2)}\simeq
 \tfrac12[\dot h_+^2+\dot h_\times^2-(\partial_z h_+)^2-(\partial_z h_\times)^2].
                                                               \tag{24}
\]

Thus the two-field action, Hamiltonian and principal polynomial are

\[
 S_{phys}^{(2)}=\frac{\sigma}{4\kappa}\sum_{a=+,\times}\int
                   [\dot h_a^2-(\nabla h_a)^2],\quad
 H_{phys}^{(2)}=\frac{\sigma}{4\kappa}\sum_a\int
                   [\dot h_a^2+(\nabla h_a)^2],              \tag{25}
\]
\[
 \det\mathcal P=\left(\frac{\sigma}{2\kappa}\right)^2
                        (\omega^2-|\mathbf k|^2)^2.          \tag{26}
\]

The Hessian is σI₂/(2κ). With the stated orientation and κ>0, σ=+ gives two healthy massless modes; σ=− gives two wrong-sign modes **relative to fixed positive-energy matter**. Simplicity alone admits both. The vacuum equations cannot distinguish an overall action sign; relative energy/probability normalization becomes physical when a matter sector is fixed. Healthy sign selection is additional physical input, not another theorem about the constraint.

Under a rotation by θ about the wave vector, h_++ih_× transforms by e^{2iθ} and its conjugate by e^{−2iθ}. Together with (22),(26) this identifies the physical helicities ±2. If the free reduced fields are quantized, q_a=h_a/√(2κ) are ordinary positive-pairing massless canonical fields for σ=+. This is not a quantization of all Plebanski sectors.

The waves are not pure coordinate artifacts: for h_+=a cos[k(t−z)], h_×=0, linearized R_0x0x=−(1/2)∂_t²h_xx=(k²/2)h_+ and R_0y0y=−R_0x0x, while Ricci is zero by (18). Linearized Riemann is gauge invariant on a flat background. This is nonzero Weyl curvature, excluded by the BF equation F=0 but allowed by F=φB in the constrained theory. The multiplier can encode curvature rather than forcing it to vanish. This identifies the dynamical change in the original first-order equations, not only in a rewritten metric action.

For a source term ∫J_a h_a, define Z_p=σ/(2κ). The derived equation and retarded susceptibility are

\[
 Z_p(\partial_t^2-\nabla^2)h_a=J_a,\qquad
 G_{ab}^R(\omega,\mathbf k)=
 \frac{\delta_{ab}}{Z_p[|\mathbf k|^2-(\omega+i0)^2]},
 \quad G_{ab}^R(t,\mathbf k)=\delta_{ab}\theta(t)
               \frac{\sin(|\mathbf k|t)}{Z_p|\mathbf k|}.     \tag{27}
\]

The kernel has a massless pole, real frequencies and positive positive-frequency spectral weight when Z_p>0. Its wave-equation inverse has retarded support; the preparation/no-incoming prescription is supplied, not a time-arrow theorem. For general localized conserved sources, use harmonic gauge and the trace reverse: □h̄_μν=−2κT_μν/σ, with the retarded wave inverse. Source conservation preserves the harmonic constraint. This avoids mistaking a nonlocal spatial TT projection of arbitrary sources for a local causal operation. Equation (27)'s simple channel is a fixed polarization/Fourier calculation.

## 7. What was and was not proved

This is a complete local **linear regular-branch** physical calculation: algebraic tangent completeness, connection elimination, canonical count, full-symbol gauge quotient, kinetic sign, helicity and response. The nonlinear identification of that branch with Palatini follows from report 01's regular holonomic reduction. A fresh full nonlinear first-order Dirac analysis across branches, quantum constraint algebra, global topology, degenerate canonical strata and nonlinear stability are not claimed. The known non-chiral canonical benchmark has 18 connection configurations, 10 first-class and 12 second-class constraints, giving (36−20−12)/2=2; see Alexandrov & Krasnov, [0809.4763](https://arxiv.org/pdf/0809.4763), §2, manuscript pp.5–6. That known result is not counted as a newly executed test.

`derive_modes.py` executes 19 exact operator/canonical/kinetic checks. `check_independent.py` executes 19 separate rational-form/rank/counterexample cases including report 01's checks. These support the displayed analytic derivations, not a claim that passing tests certifies full quantum gravity.
