# I06 — exact local physical test of the one-singlet relaxation

**Result on the declared regular gravitational component:** finite λ gives Einstein gravity plus one genuine scalar, with a positive kinetic term. For λ>0 its flat-vacuum mass squared is 4/(3λ), and its potential is nonnegative on |β|<1. There are three gravitational-sector configuration degrees of freedom, not eight and not two. Maxwell propagation can nevertheless have one metric light cone while B is nonsimple. This is a classical local result, not a UV quantum or global stability theorem.

## 1. Exact auxiliary reduction from the original first-order action

For B=starΣ+βΣ, equation D_AB=0 is

\[
 (\star+\beta)D_A\Sigma+d\beta\wedge\Sigma=0,\qquad
 (\star+\beta)^{-1}=\frac{\beta-\star}{1+\beta^2}.             \tag{14}
\]

There is no real-β singular connection block: det(star+β)=(1+β²)³. The 24-component map from metric-compatible contorsion to DΣ is invertible in a nondegenerate coframe. Splitting A=A(e)+C, the solved contorsion is

\[
 C^{IJ}=\frac{1}{2(1+\beta^2)}
 \left[\beta(e^I\partial^J\beta-e^J\partial^I\beta)
          -\epsilon^{IJ}{}_{KL}e^K\partial^L\beta\right].     \tag{15}
\]

Here ∂^Iβ=η^IJ e_J^μ∂_μβ. Thus torsion is an auxiliary response to dβ, generally with both trace and axial content; no pure-trace solution was imposed. The independent connection does not supply propagating vector/tensor ghosts.

The script derives (15) by inverting the full 24×24 equation, retaining all four independent gradient components. A second program independently expands the quadratic connection part of the original exterior action and solves its Hessian with rational arithmetic. These are different computations. After integrating B∧dC by parts, the auxiliary on-shell contribution is half its linear term:

\[
 \mathcal L_{C,on}\operatorname{vol}
   =-\tfrac12d\beta\wedge\Sigma^{IJ}\wedge C_{IJ}
   =-\frac{3}{2(1+\beta^2)}(\partial\beta)^2\operatorname{vol}_e.
                                                               \tag{16}
\]

Boundary equalities here use compact variations; no unexamined boundary mode is thrown away. From (4), the multiplier part reduces to +(λν/2)τ². Substituting (8),(10), it is −2β²/[λ(1−β²)] vol_e. The **derived** regular-branch action is therefore

\[
 \boxed{S_{red}=\frac1{2\kappa}\int\sqrt{-g}
     [R-K(\beta)(\partial\beta)^2-U_\lambda(\beta)]},\qquad
 K=\frac3{2(1+\beta^2)},\quad
 U_\lambda=\frac{2\beta^2}{\lambda(1-\beta^2)}.                \tag{17}
\]

Neither the kinetic coefficient nor this potential was chosen to obtain the result. There is one free λ, not a free potential function. The known varying-Holst kinetic term agrees with Taveras–Yunes [0807.2652](https://arxiv.org/pdf/0807.2652), Eq.(49), PDF p.7. That paper is a benchmark for the kinetic structure, not a source of our multiplier-induced potential or proof of microscopic selection.

Introduce the regular canonical field

\[
 \chi=\sqrt{\frac3{2\kappa}}\operatorname{arsinh}\beta,\quad
 S_{red}=\int\sqrt{-g}\,[R/(2\kappa)-\tfrac12(\partial\chi)^2-V(\chi)],
 \quad V=\frac{\beta(\chi)^2}{\kappa\lambda[1-\beta(\chi)^2]}.
                                                               \tag{18}
\]

This is an exact local field redefinition, not removal of χ. Its derivative is finite and nonzero on the declared chart. V≥0 for λ>0 and |β|<1; it has its only stationary point there at β=0. The divergent potential/chart inversion at |β|=1 is a finite-multiplier boundary, not evidence of another propagating mode. No global avoidance of curvature singularities or concentration at the chart boundary is proved.

## 2. Field equations and the common Minkowski linearization

Varying (17), including declared probes when present, gives

\[
 G_{\mu\nu}=K\left(\partial_\mu\beta\partial_\nu\beta
       -\tfrac12g_{\mu\nu}(\partial\beta)^2\right)
       -\tfrac12U_\lambda g_{\mu\nu}+\kappa T^{probe}_{\mu\nu},
                                                               \tag{19}
\]
\[
 2K\Box_g\beta+K'(\partial\beta)^2-U'_\lambda=0,
 \quad\text{or}\quad\Box_g\chi-V'(\chi)=0.                  \tag{20}
\]

In first-order variables the probes add their explicit B-variation to the F=φB equation; they do not alter δφ,δν or D_AB=0 because they have no multiplier/connection dependence. Equations (19),(20) include these reduced source variations; the vacuum equation F=φB is not incorrectly reused with a finite matter source.

The **same** Minkowski ē=dx,β̄=0,Ā=0,φ̄=0,ν̄=−vol solves hard and soft vacuum theories, since U(0)=U′(0)=0. At first order

\[
 G^{(1)}_{\mu\nu}[h]=0,\quad
 (\Box_\eta-M^2)\delta\beta=0,\quad
 M^2=\frac4{3\lambda},\quad
 a=A^{(1)}(h)+C(d\delta\beta),\quad \delta\tau=2\delta\beta/\lambda.
                                                               \tag{21}
\]

Scalar stress is quadratic, so it does not fake a linear metric polarization. The two tensor helicities remain those of the fully constrained metric quotient in I05. The new β is a Lorentz invariant scalar, not removable by internal Lorentz gauge; a diffeomorphism gives δβ=ξ·dβ̄=0 on this background, so its linear fluctuation is also gauge invariant. Its nonzero momentum and retarded scalar pole make it physical.

The reduced quadratic action is

\[
 S^{(2)}=\frac1{4\kappa}\sum_{a=+,\times}\int
      [\dot h_a^2-(\nabla h_a)^2]
 +\frac12\int[\dot\chi^2-(\nabla\chi)^2-M^2\chi^2].           \tag{22}
\]

There are no wrong-sign kinetic modes here for the chosen positive branch. λ<0 gives a tachyon around β=0, **not a negative-norm kinetic scalar**. Reversing the combined gravitational branch/orientation sign instead reverses gravitational/scalar kinetic terms relative to positive probe energy. Positivity rejects those signs but does not choose λ=0 or its finite positive magnitude.

The physical principal polynomial of the three geometry modes is proportional to (g^{μν}k_μk_ν)³. The scalar mass changes the pole, not the characteristic cone. Its vacuum dispersion is ω²=|k|²+M²; for λ>0 frequencies are real and no scalar tachyon occurs. The canonical retarded response is

\[
 G_\chi^R(\omega,k)=[|k|^2+M^2-(\omega+i0)^2]^{-1}.          \tag{23}
\]

It has positive positive-frequency spectral weight and causal Klein–Gordon support. Tensor response has the positive massless pole computed in I05. These are local linear propagation/energy statements, not full nonlinear quantum stability. Retarded preparation is a supplied state/boundary prescription, not an emergent time arrow.

## 3. Generic nonlinear constraint rank, beyond mode counting

The pointwise rank-19 theorem and regular τ inversion in report 01 are nonlinear. They ensure that the 19 constrained B directions are eliminated exactly and leave e plus β, rather than hiding six extra nonmetric modes in a specially chosen flat vacuum. Equation (14) eliminates all 24 connection components algebraically everywhere in this chart.

For a spacelike slice with lapse N>0 and positive spatial metric γ_ij, scalar momentum and its velocity Hessian are

\[
 p_\beta=\frac{\sqrt\gamma K}{\kappa N}
                  (\dot\beta-N^i\partial_i\beta),\qquad
 \frac{\partial p_\beta}{\partial\dot\beta}
                     =\frac{\sqrt\gamma K}{\kappa N}>0.      \tag{24}
\]

The ADM six-metric DeWitt block is invertible; adding (24) gives a seven-velocity rank-seven Legendre map (before lapse/shift). Thus there is no additional primary scalar constraint or vanishing scalar kinetic term at β=0. In canonical χ coordinates the constraints are

\[
 \mathcal H=\frac{2\kappa}{\sqrt\gamma}
       (\pi^{ij}\pi_{ij}-\tfrac12\pi^2)
       -\frac{\sqrt\gamma}{2\kappa}\,{}^3R
       +\frac{p_\chi^2}{2\sqrt\gamma}
       +\frac{\sqrt\gamma}{2}\gamma^{ij}\partial_i\chi\partial_j\chi
       +\sqrt\gamma V=0,
\]
\[
 \mathcal H_i=-2\gamma_{ij}D_k\pi^{jk}+p_\chi\partial_i\chi=0.
                                                               \tag{25}
\]

With the usual canonical brackets these four first-class constraints obey the hypersurface-deformation algebra. For the new scalar contribution a direct functional differentiation gives

\[
 \{H_\chi[N],H_\chi[M]\}
 =\int p_\chi\partial_i\chi\,\gamma^{ij}
                         (N\partial_jM-M\partial_jN);        \tag{26}
\]

the potential terms cancel. Adding the metric part and the mixed metric variations gives the total momentum constraint with the same structure function. The lapse/shift are multipliers, not independent physical pairs. Therefore on generic regular initial data

\[
 N_{geometry}=\tfrac12[2(6+1)-2(4)]=3.                      \tag{27}
\]

At linear order the explicit 14-dimensional phase matrix has constraint rows (h_xx+h_yy,π_xz,π_yz,π_zz), rank four, zero mutual bracket and four gauge directions. It agrees with (27); it is not the sole evidence for the nonlinear count. Probe degrees are separate and not included in this geometry count. The original all-variable first-order Dirac matrix on singular strata is not reconstructed. On this regular component exact holonomic/auxiliary reduction and nondegenerate Legendre transform suffice; the full-rank-potential eight-mode result of [0809.4763](https://arxiv.org/pdf/0809.4763), §3, does not apply to our rank-one potential.

## 4. Operational probe principal symbols and what is assumed

After enforcing the nineteen constraints, J_B is wedge self-adjoint, and varying the primary Maxwell action (13) gives df=0,d(J_Bf)=0. For arbitrary unconstrained B an action variation would instead use the wedge-symmetric part of J_B; no unrestricted-B hyperbolicity theorem is claimed. On the regular family J_B=*g exactly, so the principal symbol for the potential a_μ is

\[
 \mathcal M^{\nu\sigma}(k)=(g^{\alpha\beta}k_\alpha k_\beta)g^{\nu\sigma}
                                             -k^\nu k^\sigma.\tag{28}
\]

It has rank three off the null cone and rank one on it; after U(1) gauge quotient the two physical polarizations propagate on g^{μν}k_μk_ν=0. Directly the action has density (E²−B²)/2 and Hamiltonian (E²+B²)/2 in an orthonormal frame. A positive free-mode quantum pairing can consequently be chosen. This does not derive the Born rule, a full interacting quantum theory or universal geometry from probabilities alone.

For an independently calibrated clock only, take

\[
 S_\psi=-\tfrac12\int\sqrt{-g(e)}[g^{\mu\nu}\partial_\mu\psi\partial_\nu\psi
                                                   +m_\psi^2\psi^2].\tag{29}
\]

Its principal symbol is g^{μν}k_μk_ν. The common cone follows **given (13) and the shared metric choice (29)**. Maxwell's conformal operator is calculated from B, but its law (13) and the clock's use of that same normalized metric are assumptions. No observation or gauge identity derived those matter prescriptions. Maxwell alone cannot establish a universal metric for every possible probe or fix a length scale. Our fixed probes do not directly couple to β; their leading flat-vacuum response can therefore miss a genuine geometry mode. Its stress and backreaction are not absent.

## 5. On-shell nonsimple counterexample, not arbitrary off-shell B

A constant nonzero β on a flat **vacuum** metric is not a solution: U′(β)≠0. We do not use that tempting false counterexample. Instead construct exact smooth local Cauchy data in the same regular theory. Choose 0<|c|<1, λ>0, on an initial patch

\[
 \gamma_{ij}=\delta_{ij},\quad K_{ij}=H\delta_{ij},\quad
 \beta=c,\quad p_\chi=0,\quad \partial_i\beta=0,\quad
 H^2=\frac{\kappa V(c)}3=\frac{c^2}{3\lambda(1-c^2)}.         \tag{30}
\]

Set background probes to zero. The momentum constraint vanishes and the exact Hamiltonian constraint is

\[
 {}^3R+K^2-K_{ij}K^{ij}=6H^2=2\kappa V(c).                  \tag{31}
\]

These are constraint-satisfying data, not a second perturbation background or a cosmological fit. Standard local Einstein–scalar hyperbolic existence on regular smooth data yields a local development; our potential is smooth near c. Constraint propagation follows from (25),(26)/Bianchi and the scalar equation. See the primary Cauchy analysis [Choquet-Bruhat–Isenberg–Pollack, gr-qc/0611009](https://arxiv.org/abs/gr-qc/0611009). Only its local theorem is invoked, not a new global existence proof. Pulling back through the regular reduction gives a local solution of the first-order family with finite multipliers. At the initial slice Q_B=24c vol≠0, hence B is genuinely nonsimple. V_B=(c²−1)vol≠0 and J_B=*g; the metric, probe cone and kinetic terms are regular. The scalar will evolve because V′(c)≠0; stationarity was never assumed.

This is an exact local existence counterexample to **full simplicity being necessary for healthy local Lorentzian propagation** in this family, conditional on the explicit positive branch/probe law. It is not a claim of all-time stability, asymptotically flat finite total energy of the constant patch, or an evaporating/cosmological model. The independent rational example c=1/3,λ=2,κ=1 gives V=1/16,H²=1/48 and satisfies (31) exactly; these are unit choices, not fits.

## 6. Constraint preservation, low-energy sectors and limits

The nineteen constraints are imposed algebraically by the retained multipliers and hold identically under regular reconstruction. Their preservation is not a dynamical attraction from arbitrary BF data. Full simplicity β=0 is different: for the β-independent probe choices, β=0 and p_β=0 form an exactly invariant subspace of (19),(20), since U′(0)=0. They yield GR with these probes for **every** finite λ. They are an optional consistent truncation, not gauge constraints of the soft theory; their mutual canonical bracket is nonzero.

Small departures satisfy a massive wave equation for λ>0 and oscillate. There is no damping or attractor selecting β=0. λ→0⁺ sends M→∞; the hard theory is a singular heavy-mode limit, not a finite regular field redefinition. At energies well below M, or in the exactly unexcited β=0 preparation, a healthy two-tensor description does not identify λ. If instead one demands **exactly two total geometry modes at generic configurations**, every finite λ fails within this rank-one family and λ=0 is selected. This is a sharply stronger criterion than the existence of a two-mode sector.

At quadratic order about β=0 the tensor/scalar blocks are diagonal. A typed linear retained-mode projection and retarded scalar Schur elimination both leave the tensor block unchanged because the cross block is zero. This does not erase the complementary scalar's independent initial data, its pole or nonlinear backreaction. No one-loop scalar determinant or observer partial trace is being calculated.

For completeness, on B=Σ+γstarΣ the same auxiliary calculation gives

\[
 S_{top\ chart}=\frac1{2\kappa}\int\sqrt{-g}
 [\gamma R+\frac{3\gamma}{2(1+\gamma^2)}(\partial\gamma)^2
                  +\frac{2\gamma^2}{\lambda(1-\gamma^2)}].   \tag{32}
\]

For γ>0 this agrees with (17) under g_E=γg,β=1/γ; the displayed expression is regular algebraically at γ=0. At the topological flat point its quadratic action has γR^(1)+2γ²/λ, with **no TT kinetic term**. Linear equations enforce (η_μν□−∂_μ∂_ν)γ=0 and R^(1)+4γ/λ=0. A nonzero Fourier mode has γ=0 and R^(1)=0, while TT variations have zero quadratic kinetic operator. This is not a healthy graviton theory or proof that no nonlinear variables exist there. The generic three-mode theorem requires a nonzero effective Einstein coefficient and does not extend across that singular stratum. Degenerate V=0 configurations have no inverse-defined probe action. Their nonlinear mode counts remain unresolved, explicitly outside the regular result.
