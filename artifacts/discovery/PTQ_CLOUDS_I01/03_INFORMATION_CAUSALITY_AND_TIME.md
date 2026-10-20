# Information, causality and attempted time-arrow derivation

The same calculation gives observer-relative thermality with global purity. It does not produce fundamental destruction of information or an entropy-producing arrow.

## 1. Exactly what is lost, and to whom?

For the mode representation (02:17), let r=q²∈(0,1). Summing the actual probabilities p_n=(1−r)rⁿ gives

\[
 \bar n=\frac r{1-r},\quad
 S_R=-\log(1-r)-\frac r{1-r}\log r,\quad
 \operatorname{Tr}\rho_R^2=\frac{1-r}{1+r}.                    \tag{24}
\]

The global pair is pure: S_RL=0, purity 1. The quantum mutual information is I(R:L)=2S_R. Cross-wedge correlations include

\[
 \langle b_Rb_L\rangle_{\Psi_q}=\frac q{1-q^2},\quad
 \langle b_R^\dagger b_R\rangle_{\Psi_q}=\frac{q^2}{1-q^2}.      \tag{25}
\]

An observer limited to the right algebra cannot measure b_L or the joint operator b_Rb_L. The restriction is noninjective on global states. This is loss of access to distinctions and correlations, rather than a dynamical change to the global state.

An explicit alternative global state is

\[
 \rho_{\rm deph}=\sum_np_n|nn\rangle\langle nn|.               \tag{26}
\]

It has exactly the same right and left marginals as Ψ_q, but 〈b_Rb_L〉=0, S_RL=S_R and I(R:L)=S_R. Therefore **no measurement confined to one marginal can distinguish (17) from (26)**. The two states are globally distinguishable. This is a counterexample to inferring global information destruction from a thermal marginal. Conversely a thermal marginal alone cannot prove global purity. A global dephasing channel can itself have a unitary environment dilation; (26) is not proof that any fundamental nonunitary law exists.

At q=1/2 the untruncated pair gives n̄=1/3, purity 3/5 and cross correlation 2/3; finite sums independently reproduce these values and (24). The d=2 occupation diagnostic instead gives 〈b_Rb_L〉=2/5 for the pure truncated pair and 0 for its dephased state, with global purities 1 and 17/25. The distinct numbers are the stated truncation, not a conflict between theories. At q=0 the pair is a product, the local entropy is zero and the distinction disappears.

These are finite-mode information measures. **No finite S_R, S_RL or I(R:L) of adjacent unregulated continuum wedges is claimed by (24).** The global Fock vacuum is pure, but the continuum wedge algebra has no intrinsic trace giving a finite ordinary von Neumann entropy. The channel calculations certify the regulated domain only. Continuum relative-entropy or separated-region mutual-information calculations were not performed.

## 2. Thermality requires the state, not just a cut

A counterfactual state tests the dependence on the vacuum while keeping the same field equation, algebra and accelerated worldline. It is not an additional observer protocol or a replacement state in the main fixed-vacuum comparison. Choose one normalized right-moving particle

\[
 |1_f\rangle=\int_0^\infty dk\,f(k)a_k^\dagger|0_M\rangle,
 \quad f(k)=2\ell\sqrt{k}e^{-\ell k},\quad \ell>0,
 \quad \int_0^\infty |f(k)|^2dk=1.                            \tag{27}
\]

Its vacuum-to-one-particle wave function is

\[
 F(U)=\int_0^\infty\frac{dk}{\sqrt{4\pi k}}f(k)e^{-ikU}
 =\frac{\ell}{\sqrt\pi(\ell+iU)}.                            \tag{28}
\]

The additional two-point function is F(U)F(U')*+F(U)*F(U'). For the measured derivative, its smooth excess at coincidence is

\[
 \Delta C(\tau,\tau)=2|\dot F(U(\tau))|^2
 =\frac{2\ell^2e^{-2a\tau}}
 {\pi[\ell^2+a^{-2}e^{-2a\tau}]^2}.                          \tag{29}
\]

This is nonstationary, so cannot satisfy the stationary boost KMS condition. For aℓ=1 its values at aτ=0 and log2 are respectively a²/(2π) and 8a²/(25π). An independent numerical positive-frequency integral of dF/dτ verifies the excess; the normalization and symbolic derivatives are separately checked. Therefore restriction to a causal algebra **does not thermalize every pure state**. Vacuum entanglement and boost analyticity, alongside the chosen access/time flow, are essential in the positive result. A boundary alone does not supply them.

## 3. Causal response and time reversal

For □=∂_t²−∂_x², the real inverses are

\[
 G_{\rm ret}(t,x)=\tfrac12\Theta(t)\Theta(t^2-x^2),\qquad
 G_{\rm adv}(t,x)=\tfrac12\Theta(-t)\Theta(t^2-x^2),\quad
 \Box G_{\rm ret/adv}=\delta(t)\delta(x).                     \tag{30}
\]

Distributional differentiation of G_ret in null coordinates checks the normalization; it is Θ(U)Θ(V)/2 and □=4∂_U∂_V, while δ(U)δ(V)=δ(t)δ(x)/2. Its causal support and (01:4) give the Kubo convention G_ret=iΘ(t)〈[φ(t,x),φ(0,0)]〉. For actual current observables differentiate the endpoints and smear. Spacelike commutators vanish even though spacelike vacuum correlations are nonzero. The latter do not transmit a controllable signal across the wedge boundary.

A classical admissible source J added as +∫Jφ gives □δφ=J and δφ_ret=G_ret*J after imposing vanishing incoming response. A source for currents can equivalently be written with J=−∂_μf^μ, which is infrared admissible. This inverse is different from the positive-frequency Wightman kernel used for noise and transition rates.

The explicit transformation is

\[
 G_{\rm ret}(-t,x)=G_{\rm adv}(t,x),\quad
 J_T(t,x)=J(-t,x),\quad
 (\delta\phi_{\rm ret}[J])_T=\delta\phi_{\rm adv}[J_T].         \tag{31}
\]

The parent equation and vacuum are T invariant. A fixed retarded preparation prescription is not transformed into itself. Selecting a retarded solution is the specified operational causal order of preparation and response; it does not derive an entropy arrow from the differential equation. PT additionally exchanges the two wedges; a causal-access cut is not the PT-even Hilbert eigenspace. Positive pairing and local commutativity are supplied/derived properties of this ordinary QFT, not consequences of idempotence.

## 4. Try to obtain an irreversible arrow

First attempt: infer entropy growth from the thermal restriction. This fails exactly. Boosts preserve A_R and act by automorphisms α_η, with inverse α_−η. The vacuum-restricted state is stationary. In the regulated representation,

\[
 B_\Omega=\Omega(N_R-N_L),\quad B_\Omega\Psi_q=0,\qquad
 \rho_R(\tau)=e^{-i\omega N_R\tau}\rho_R e^{i\omega N_R\tau}
 =\rho_R,\quad \frac{dS_R}{d\tau}=0.                         \tag{32}
\]

Thermal equilibrium has nonzero entropy and fluctuations but no entropy production in this stationary calculation. Noninjectivity of a description map is not a proof that a physical process has erased information. Nor is the choice of a boost clock the invention of a new irreversible evolution. For a fixed wedge, ordinary Minkowski translations do not preserve the accessible algebra; substituting H_M for the boost flow in a supposed closed wedge dynamics would change this question.

Second attempt: obtain a universal arrow from reversible laws and restriction alone. A limited no-go is immediate. Let the allowed trajectories be closed under T, let F be an observer entropy/state functional invariant under T, and suppose F is differentiable and nondecreasing on every allowed forward trajectory. For a trajectory γ(t), its allowed reverse is γ_T(t)=Tγ(−t), so

\[
 \frac d{dt}F[\gamma(t)]\geq0,\qquad
 \frac d{dt}F[\gamma_T(t)]=-\left.\frac d{du}F[\gamma(u)]\right|_{u=-t}\geq0.
                                                               \tag{33}
\]

Applying both at corresponding points forces dF/dt=0. A universal strict arrow therefore requires a restriction of allowed histories/preparations, a boundary condition, or an additional irreversible physical law. This does **not** forbid entropy increase in a subsystem for a specially prepared nonequilibrium state, or an entropic arrow for typical states under a low-entropy boundary condition. Those hypotheses are absent from the claimed universal deduction and must be supplied explicitly. In the current thermal example, the stronger simple result (32) already defeats the attempted arrow.

## 5. Locate every possible asymmetry

| Ingredient | Supplied or derived? | What it does / does not explain |
|---|---|---|
| Vacuum positive-frequency spectrum | Supplied state, T invariant | Its analytic correlations produce (8) and KMS with boosts; no low-entropy time boundary chosen |
| Right-wedge access and eternal acceleration | Supplied causal/operational restriction | Makes left/joint data inaccessible and fixes proper-time scaling; not entropy production |
| Initial vacuum entanglement | Derived from global preparation/analytic mode relation | Makes marginals mixed while preserving purity; not a collapse event |
| Retarded source prescription | Supplied initial-response condition | Selects future response; T gives advanced, not the same prescribed experiment |
| Initially prepared detector and χ | Supplied apparatus operation | Finite-time transition probabilities and switching transients; no autonomous irreversible relaxation has been proved |
| Born/Markov/secular reduction, repeated resets | NOT used | Would be additional assumptions for a forward semigroup; cannot be silently inferred from (13) |
| A low-entropy cosmological boundary | NOT used | The calculation provides no origin for it |

The partial trace at one instant is CPTP for arbitrary inputs. An autonomous reduced **dynamical** CPTP map for arbitrary initially correlated preparations does not follow from that fact. Under the declared decoupled boost generator it is simply unitary on R even when the initial state is entangled. Under a different nonclosed subsystem evolution, complementary history must be retained.

For clarity, the exact amplitude-level block identity for any valid Hilbert P,Q and self-adjoint parent H is

\[
 \psi_Q(t)=e^{-iH_{QQ}(t-t_0)}\psi_Q(t_0)
 -i\int_{t_0}^t ds\,e^{-iH_{QQ}(t-s)}H_{QP}\psi_P(s),
\]
\[
 i\dot\psi_P=H_{PP}\psi_P+
 H_{PQ}e^{-iH_{QQ}(t-t_0)}\psi_Q(t_0)
 -i\int_{t_0}^t ds\,H_{PQ}e^{-iH_{QQ}(t-s)}H_{QP}\psi_P(s).
                                                               \tag{34}
\]

These follow by solving the Q equation, provided the Q-block has the self-adjoint realization needed to generate its displayed unitary group and the blocks share the required domains. They hold literally for the finite matrices in our regulator; an arbitrary compression of an unbounded self-adjoint operator does not automatically meet those hypotheses. Dropping the inhomogeneous term assumes complementary initial data; dropping/approximating the integral assumes away memory. For (02:23) H=B_Ω, both cross terms vanish, so no nonzero memory kernel is manufactured. The thermal information is already in the correlated state. A density-matrix partial trace is not (34); the exact Gaussian density-kernel relation (02:21) is the available relation for this model. This both respects the Particles/D01/D02 distinction and prevents carrying their nonzero mixings into a different parent by analogy.

## 6. Independent counterexamples and scope

1. Thermal marginal versus pure global state: (17),(24) disprove thermality ⇒ global nonunitarity.
2. Same marginal, different global purity/correlation: (25),(26) disprove restricted statistics ⇒ global information reconstruction or global destruction.
3. Nonstationary one-particle wavepacket: (27)–(29) disprove accessibility alone ⇒ KMS for every pure state.
4. Thermal state with Ṡ_R=0: (32) disproves thermality ⇒ time arrow.
5. Retarded transforms to advanced: (31) disproves a fixed retarded inverse being fixed by T invariance of the parent.
6. Postselected vacuum versus thermal trace and zero boost Schur cross block: (02:18)–(23) disprove equality of Hilbert projection, elimination and observer restriction.

These are analytic examples with independently checked representative calculations. There is no black-hole metric, evaporation, radiation recovery channel, dynamical geometry or measurement-outcome selection in this calculation.
