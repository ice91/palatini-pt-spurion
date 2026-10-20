# Exact observer reduction and measured response

The computations below use the fixed parent and two protocols of report 01. They reproduce a nontrivial known physical effect without assuming a thermal reduced state.

## 1. Derive the detector correlator from vacuum modes

Integrating the two chiral positive-frequency mode integrals in (01:2), up to an irrelevant infrared constant, gives the formal generating distribution

\[
 W(x,x')=-\frac1{4\pi}\log\{\mu^2[(\Delta x)^2-(\Delta t-i0)^2]\}.
                                                               \tag{6}
\]

This formula is used only after admissible derivatives. In null coordinates its two logarithms are log(U−U'−i0) and log(V−V'−i0), with the boundary phases fixed by positive frequency. On z_a(τ),

\[
 U(\tau)=-a^{-1}e^{-a\tau},\quad V(\tau)=a^{-1}e^{a\tau},
 \quad (\Delta x)^2-(\Delta t)^2=-4a^{-2}\sinh^2(as/2),
 \qquad s=\tau-\tau'.                                         \tag{7}
\]

Differentiate each logarithm at both endpoints. The result, as a proper-time boundary distribution, is

\[
 C_a(s)=\langle\dot\phi(\tau)\dot\phi(\tau')\rangle
       =-\frac{a^2}{8\pi}\frac1{\sinh^2[a(s-i0)/2]},\qquad
 C_0(s)=-\frac1{2\pi(s-i0)^2}.                                \tag{8}
\]

The i0 in (8) denotes the stationary analytic boundary value, not an assertion that finite coordinate ε pulls back to a constant proper-time regulator. Taking the distributional boundary correctly matters. Direct differentiation of the two null logarithms and the a→0 limit are exact symbolic checks. Neither the Gibbs law nor a temperature entered this calculation.

For switching χ, first-order perturbation theory gives

\[
 \mathcal F_\chi(E)=\int d\tau\,d\tau'\,
 \chi(\tau)\chi(\tau')e^{-iE(\tau-\tau')}C_a(\tau-\tau').         \tag{9}
\]

Positive type of the vacuum correlator guarantees nonnegative (9). This operational statement needs the probe preparation and coupling in (01:5), not just an observer label.

## 2. Calculate the spectrum, including spontaneous response

Set β=2π/a and define the stationary rate with coupling/matrix factors suppressed:

\[
 \mathcal R_a(E)=\int_{\mathbb R}ds\,e^{-iEs}C_a(s).
                                                               \tag{10}
\]

The poles of the boundary integrand are s_n=inβ+i0, n∈Z. Near any pole, C_a(s)=−[2π(s−s_n)²]⁻¹+regular, so

\[
 \operatorname{Res}_{s_n}e^{-iEs}C_a(s)
       =\frac{iE}{2\pi}e^{-iEs_n}.                            \tag{11}
\]

For E>0 close downwards (clockwise) and sum n=−1,−2,…: R_a(E)=EΣ_{m≥1}e^{−mβE}. For E<0 close upwards and include n=0: R_a(E)=−EΣ_{n≥0}e^{nβE}. Thus

\[
 \boxed{\mathcal R_a(E)=\frac{E}{e^{\beta E}-1}},\quad
 \mathcal R_a(0)=\frac1\beta,\quad
 \mathcal R_0(E)=-E\Theta(-E).                                \tag{12}
\]

The value at E=0 is a continuous limit, independent of a convention for Θ(0). The excitation spectrum is positive, the de-excitation spectrum includes spontaneous emission, and

\[
 \frac{\mathcal R_a(E)}{\mathcal R_a(-E)}=e^{-\beta E},\qquad
 T_{\rm proper}=\frac{a}{2\pi}.                               \tag{13}
\]

No equilibrium detector state was inserted. A Planck factor in a transition rate does not say that the Minkowski stress tensor is a thermal gas: 〈:T_μν:〉=0 in the same vacuum. The detector's agent supplies acceleration/work. Energy bookkeeping for a fully dynamical accelerator is outside the weak prescribed-probe calculation.

Independent numerical calculation deforms (10) to z=t−iδ with 0<δ<β, including the factor e^{−iE(t−iδ)}. At δ=β/2 the kernel becomes a² sech²(at/2)/(8π); at δ=β/3 it is a different nonsingular complex kernel. Ten 45-digit quadratures, E/a∈{−1,−1/4,0,1/4,1}, agree with (12) to better than 10⁻³⁵ absolute error. This tests the Fourier transform rather than substituting the answer into itself; finite samples supplement the contour proof.

## 3. KMS is an analytic ordering relation

Let F(z)=−a²/[8πsinh²(az/2)]. It is analytic for −β<Im z<0. Its top boundary is C_a(s); its bottom boundary is C_a(−s), the reversed operator ordering. Imaginary periodicity F(z−iβ)=F(z) supplies this relation only with these opposite boundary limits. Simply setting a real-time function equal to itself after an imaginary shift would omit the distributional ordering.

This derives the two-point KMS property measured on the orbit. The algebra-wide vacuum wedge KMS statement is an **ESTABLISHED EXISTING RESULT**, not inferred from one detector alone: for the local Gaussian current theory, the vacuum restricted to A_R is KMS at inverse boost temperature 2π, with proper-time scaling β=2π/a. Analytic continuation of smeared wedge correlators and Wick/Weyl reconstruction supply its free-field form. The general algebraic theorem and its scope are described in Crispino–Higuchi–Matsas, §II.9, [author text](https://arxiv.org/html/0710.5373v1). We invoke that established extension; the executed symbolic tests verify the displayed two-point identities, not a new proof of the von Neumann algebra theorem.

Equation (12) also yields a physical connection between fluctuations and causal response:

\[
 \mathcal R_a(-E)-\mathcal R_a(E)=E,\qquad
 \mathcal R_a(-E)+\mathcal R_a(E)=E\coth(\beta E/2).             \tag{14}
\]

The antisymmetric spectrum is state independent for this current commutator; the symmetric noise is thermal. With Fourier convention (10), S_comm(E)=R_a(E)−R_a(−E)=−E and S_sym(E)=[R_a(E)+R_a(−E)]/2. These fix the normalization of the fluctuation–response relation rather than importing a phenomenological noise term. Causality comes from the original commutator support, not from the Planck factor.

## 4. Finite-duration response is not automatically thermal

For χ_T(τ)=exp[−τ²/(2T²)], Fourier inversion of C gives

\[
 \mathcal F_{\chi_T}(E)=T^2\int d\nu\,
       \mathcal R_a(\nu)e^{-T^2(E-\nu)^2},\qquad
 \frac{\mathcal F_{\chi_T}(E)}{\int\chi_T^2d\tau}
 =\frac1{\sqrt\pi}\int dx\,e^{-x^2}\mathcal R_a(E+x/T)
 \xrightarrow{T\to\infty}\mathcal R_a(E).                     \tag{15}
\]

The inertial finite-time excitation also need not vanish: replace R_a(ν) by −νΘ(−ν) in (15). Switching injects nonstationary spectral content. Executed integrations at a=1,E=1/2,T=2,5,15,40 verify positivity and convergence; the last error is below 10⁻⁴. These are dimensionless mathematical illustrations, not proposed empirical data. Exact detailed balance is a stationary limit, not a universal finite-time thermometry identity.

## 5. Mode-pair derivation: where the missing correlations go

This section is a **controlled boost-frequency representation** of protocol II. Continuum diagonal modes are distributions. A frequency box/occupation regulator makes the following oscillator equations literal; it is not asserted to preserve exact locality or Lorentz covariance at finite cutoff. A general finite-band wavepacket mixes frequencies and need not have the exact single-frequency Gibbs form below. The continuum result (8)–(13) was derived independently of such a regulator.

For one positive boost frequency Ω, a right null mode behaves as (−aU)^{iΩ} on U<0. Positive Minkowski frequency means analyticity in the lower U half-plane. Continuing from U<0 to U>0 gives log(−U)=log U+iπ and hence the relative magnitude q=e^{−πΩ}. Left time orientation turns its partner into a creation component. Fix its irrelevant phase so q>0. Canonical annihilators are

\[
 d_R=\frac{b_R-qb_L^\dagger}{\sqrt{1-q^2}},\quad
 d_L=\frac{b_L-qb_R^\dagger}{\sqrt{1-q^2}},\quad 0<q<1.         \tag{16}
\]

They preserve bosonic commutators. Solving d_RΨ=d_LΨ=0 gives c_{n+1}=qc_n, so

\[
 |\Psi_q\rangle=\sqrt{1-q^2}\sum_{n\geq0}q^n|n_Rn_L\rangle,
 \qquad \rho_R=(1-q^2)\sum_{n\geq0}q^{2n}|n_R\rangle\langle n_R|.
                                                               \tag{17}
\]

Thus the Boltzmann weight q²=e^{−2πΩ}=e^{−βω}, ω=aΩ, follows from global positive frequency and accessibility, not from choosing a thermal ρ_R. This is a representation of the standard analyticity mechanism. It is not a normalizable tensor-product expression for the entire unregulated Minkowski vacuum.

For a literal d-level occupation calculation use Ψ_d=Z_d⁻¹/²Σ_{n=0}^{d−1}qⁿ|nn〉, Z_d=(1−q^{2d})/(1−q²). Its partial trace is exactly the finite Gibbs diagonal distribution. Truncated b matrices obey [b,b†]=I−d|d−1〉〈d−1|; no false finite-dimensional canonical commutator is used. At d=3,q=1/2, the exact output is diag(16,4,1)/21. This is a regulator check, not an exact numerical discretization of all continuum wedge modes.

## 6. Three different operations and an exact Gaussian relation

**Observable restriction:** ω_R(A)=ω_M(A), A∈A_R. It changes which questions can be asked, not the global state. In a finite factorization it is represented by Tr_Lρ. With K_n=I_R⊗〈n_L|,

\[
 \operatorname{Tr}_L\rho=\sum_nK_n\rho K_n^\dagger,\qquad
 \sum_nK_n^\dagger K_n=I_{RL}.                               \tag{18}
\]

This is a completely positive trace-preserving channel on all trace-class RL states, including initially entangled inputs. Its adjoint A↦A⊗I is a unital *-embedding. In the continuum this inclusion remains well defined; a canonical trace-class partial trace across the sharp horizon does not.

**Hilbert projection:** P_n=I⊗|n〉〈n|. One selective outcome K_nρK_n† is CP and trace nonincreasing; normalized conditioning is nonlinear. For (17), postselecting n_L=0 gives the right vacuum with success probability 1−q², not ρ_R. At finite d the probability is 1/Z_d. Exactly,

\[
 \operatorname{Tr}_L\sum_nP_n\rho P_n=\operatorname{Tr}_L\rho,
                                                               \tag{19}
\]

but the global dephasing in (19) changes cross correlations. No measurement is required to restrict an algebra. The exact Choi matrix of (18) at d_R=d_L=3 has eigenvalues 3 (multiplicity 3) and 0 (multiplicity 24), and trace-preserving Kraus completeness is checked. An independent entangled-ancilla input check verifies positivity and unit trace. These statements concern the channel, not a claim that arbitrary correlated open-system preparations define a CP autonomous dynamical map.

**Schur integration:** the same oscillator pair has a real Gaussian wave function

\[
 \Psi(x,y)=\pi^{-1/2}e^{-\frac12[A(x^2+y^2)-2Cxy]},\quad
 A=\frac{1+q^2}{1-q^2},\ C=\frac{2q}{1-q^2},\ A^2-C^2=1.
                                                               \tag{20}
\]

Actual Gaussian integration gives

\[
 \rho_R(x,x')=\frac1{\sqrt{\pi A}}
 e^{-\frac A2(x^2+x'^2)+\frac{C^2}{4A}(x+x')^2}.               \tag{21}
\]

The diagonal precision is the Gaussian Schur complement A−C²/A=1/A. But the off-diagonal term is essential:

\[
 \langle x^2\rangle=A/2,\quad
 \langle p^2\rangle=-\int dx[\partial_x^2\rho_R(x,x')]_{x'=x}=A/2,
 \quad \nu=A/2>1/2\quad(q>0).                               \tag{22}
\]

Replacing (21) by a pure wave function inferred from its diagonal would instead give p²=1/(2A), losing its mixedness. Two independent numerical y-integrals and a separate precision-matrix calculation verify (21)–(22). There is an exact Gaussian Schur identity inside partial tracing, but it is not dynamical Feshbach elimination and does not license discarding the doubled density kernel.

For the boost evolution of this pair, B_Ω=Ω(N_R−N_L), not H_M. Take P=I_R⊗|0_L〉〈0_L|, Q=I−P as an algebraic diagnostic. PB_ΩQ=0; with Im z≠0 the complementary resolvent exists, and

\[
 D_{\rm eff}(z)=P(z-B_\Omega)P-
 P(z-B_\Omega)Q[Q(z-B_\Omega)Q]^{-1}Q(z-B_\Omega)P
 =P(z-B_\Omega)P.                                           \tag{23}
\]

The thermal ρ_R still exists. Hence a nonzero dynamical Schur correction is not required for this thermality; state entanglement carries its information. B_Ω has no global lower bound and is not the positive Minkowski Hamiltonian. The pair is not a closed subsystem under H_M. Equations (21) and (23) concern different operators and domains, and no equality of the three types of reduction has been asserted.

Executed evidence: `evidence/derive_observer.py` / `symbolic_results.json`; independent `evidence/test_independent.py` / `independent_results.json`. Equations displayed here are analytic derivations. Passing finite tests does not prove continuum factorization or a new causal theorem.
