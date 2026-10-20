# I02 — adversarial tests and explanatory decision

**Completed finding:** causal access plus a specified vacuum determine quantitative modular information–energy relations. They do not determine a new metric equation, a universal temperature for all states, or an entropy-producing arrow. The known gravitational interface is conditional on additional horizon-entropy/equilibrium premises.

## 1. Explicit state-dependent counterexamples

### Energy is not a complete information description

For report 01's noncommuting ρ_ε and its number-dephased state \(\mathcal P(\rho_\epsilon)\), the diagonal is identical. Hence ΔK is identical, while

\[
 D(\rho_\epsilon\Vert\sigma)
 -D(\mathcal P\rho_\epsilon\Vert\sigma)
 =S(\mathcal P\rho_\epsilon)-S(\rho_\epsilon)>0
 \quad(\epsilon\ne0).                                      \tag{36}
\]

Strict entropy increase under pinching follows from its relative-entropy identity D(ρ∥Pρ)=S(Pρ)−S(ρ)>0 for a non-diagonal ρ; the positivity proof is in report 01. At ε=0.02, ΔK≈0.02772588722 for both, but D≈0.002228551160 versus 0.001279893247 nats. This is an actual finite-mode demonstration: even a fixed modular-energy expectation does not determine relative entropy. It is not inferred that arbitrary regulator matrices have the same full continuum stress tensor.

There is also a continuum stress counterexample in the calculated family. Coherent preparations +λ and −λ have the **same entire stress expectation**, since (01:11) is quadratic, but opposite local current means. A fixed smeared current can distinguish them; this global sign symmetry is not gauged away in our declared observable algebra. Unitary covariance of relative entropy and Weyl composition give

\[
 D_R(\omega_{+\lambda}\Vert\omega_{-\lambda})
 =D_R(\omega_{2\lambda}\Vert\omega_0)
 =4D_R(\omega_\lambda\Vert\omega_0)>0.                      \tag{37}
\]

Their relative entropies to the vacuum are equal, not different; (37) concerns the mutual comparison. Thus stress/modular energy alone does not specify the state or all physical distinctions. This is a comparison of preparations in one theory, not different universes.

### Modular flow need not be bare geometric evolution for every state

For a coherent reference state, modular theory and (9) give exactly

\[
 \sigma_s^{(\lambda)}(A)=U_\lambda
       \sigma_s^{(0)}(U_\lambda^\dagger A U_\lambda)U_\lambda^\dagger.
                                                               \tag{38}
\]

If B_s denotes the geometric boost action, its action on a current is

\[
 \sigma_s^{(\lambda)}(j)=B_sj+j_{\rm cl}-B_sj_{\rm cl}.       \tag{39}
\]

The last term is a state-dependent c-number displacement. A nonzero compact packet is not boost stationary. The flow can be put into vacuum form on appropriately displaced currents, but it is not a new coordinate metric on the original currents. Existence of modular flow alone therefore does not imply a geometric boost, a universal proper-time temperature, or curvature. I01's explicitly nonstationary one-particle correlator supplies a second existing control against state-independent KMS assertions.

The coherent path also falsifies finite ΔS=ΔK: every local-unitary regulator has ΔS=0 and finite ΔK>0. The first law is a differential identity around its fixed reference. Neither that identity nor D≥0 says that entropy is produced along physical time.

## 2. Secondary finite-switching calculation: same detector, actual excess

Keep I01's derivative-coupled detector, trajectory z_a(τ)=a⁻¹(sinh aτ,cosh aτ), initially prepared detector ground state independent of the field, coupling normalization and Gaussian switching χ_T=e^{−τ²/(2T²)}. Compare only the field preparation: vacuum versus I01's normalized right-moving one-particle state (f_k=2ℓ√k e^{−ℓk}). No new detector or fitted data is introduced.

Its extra current correlator is a smooth finite-rank kernel,

\[
 \Delta C(\tau,\tau')=g(\tau)g(\tau')^*+g(\tau)^*g(\tau'),
 \quad g(\tau)=\frac d{d\tau}\frac{\ell}{\sqrt\pi[\ell+iU(\tau)]},
 \quad U=-a^{-1}e^{-a\tau}.                                 \tag{40}
\]

Substitution into the actual response integral gives the exact leading-order excess

\[
 \boxed{\Delta\mathcal F_{\chi}(E)
 =|I_\chi(E)|^2+|I_\chi(-E)|^2\geq0},\quad
 I_\chi(E)=\int d\tau\,\chi(\tau)e^{-iE\tau}g(\tau).          \tag{41}
\]

It is even in E. Thus the state changes noise/transition probabilities while leaving the field commutator and causal support unchanged. Positivity here is derived from the particular one-particle kernel; it is not a claim that every excited state's response excess must be positive for every protocol.

The vacuum background is calculated separately from the I01 spectrum,

\[
 \mathcal F_{0,T}(E)=T^2\int d\nu\,
       \frac{\nu}{e^{2\pi\nu/a}-1}e^{-T^2(E-\nu)^2}.
                                                               \tag{42}
\]

Both terms use the same switch, so switching transients have not been called a temperature shift. For a=ℓ=1, E=1/2:

| T | Vacuum F_0 | State-dependent ΔF | One-particle F_1 |
|---|---:|---:|---:|
| 2 | 0.182692222092 | 0.546051699926 | 0.728743922018 |
| 5 | 0.238665263122 | 0.697038359884 | 0.935703623006 |

These are dimensionless leading **response functions**, not unit-size probabilities: multiply by λ_detector²|m_eg|² and stay in the perturbative domain. Different expectations imply statistical discriminability over repeated identically prepared probes, not perfect identification from a single click or a tomographic reconstruction of the field state. Finite-duration data need not obey the infinite-time KMS rate ratio.

There is an independent analytic/convolution check. At a=ℓ=1,

\[
 g(\tau)=\frac1{2\sqrt\pi(1+i\sinh\tau)},\quad
 I_\infty(\nu)=\sqrt\pi\,\frac{\nu e^{-\pi\nu/2}}{\sinh(\pi\nu)},
 \quad I_\infty(0)=1/\sqrt\pi.                              \tag{43}
\]

To derive (43), set y=e^{−τ}; the integral becomes −i/√π∫₀∞ y^{iν}(1−iy)⁻²dy. Euler's beta integral with the stated complex branch gives e^{−πν/2}|Γ(1+iν)|²/√π; |Γ(1+iν)|²=πν/sinh πν. A Gaussian convolution then gives

\[
 I_{\chi_T}(E)=\frac1{\sqrt{2\pi}}\int dy\,e^{-y^2/2}
                         I_\infty(E+y/T).                  \tag{44}
\]

An independently implemented frequency convolution agrees with direct time integration to better than 10⁻³⁵ for both signs and durations. The no-switch excess is finite,

\[
 \Delta F_\infty(E)=2\pi E^2\frac{\cosh(\pi E)}{\sinh^2(\pi E)},
 \qquad \Delta F_\infty(0)=2/\pi,\quad(a=\ell=1),             \tag{45}
\]

while the vacuum's total unwindowed response diverges with observation duration. Formula (45) concerns only the excess, not a finite eternal total probability. Its duration-normalized contribution vanishes at long time. A localized excitation can be detectable in finite time without producing a different stationary thermal rate.

## 3. Identical accessible restrictions: an exact impossibility result

If ω_1|_M=ω_2|_M, every strictly M-local protocol has identical outcome probabilities. Proof: couple an independently prepared probe in the same state τ to M, using unitaries/instruments in M⊗B(H_probe). Each history h induces a positive effect E_h∈M (including adaptive steps), and

\[
 p_1(h)=\omega_1(E_h)=\omega_2(E_h)=p_2(h).                  \tag{46}
\]

Equivalently, tensoring equal restricted states with the same τ and applying the same local CP maps cannot make them unequal. Conditional probabilities also agree whenever the conditioning history has nonzero probability. Access to complementary operations, initially differently correlated probes, or observable evolution outside M changes the protocol and is outside this assertion.

An actual continuum example is a coherent Weyl unitary V_L supported strictly in W_L: V_L∈M'. The global pure states Ω and V_LΩ differ and have different left stress/mean fields, but

\[
 \langle V_L\Omega,A V_L\Omega\rangle=\langle\Omega,A\Omega\rangle
 \quad(A\in M),\qquad D_R(\omega_{V_L}\Vert\omega_0)=0.       \tag{47}
\]

No right-wedge switching, gap or sequence of local detectors can distinguish them. This complements (41), where the right restrictions really differ. I01's entangled/dephased finite pair provides another diagnostic; an independent nonprojective POVM check gives equal local probabilities. Neither result permits an observer to infer global purity, global destruction or recovery from a thermal restriction alone.

## 4. Scientific classification and explanatory compression

| Epistemic class | Finding |
|---|---|
| PROVED EXISTING PHYSICS | Vacuum wedge modular boost flow; Araki positivity and restriction monotonicity; differential entanglement first law; coherent-state information–boost-energy identity; ordinary local indistinguishability |
| NEWLY EXECUTED DERIVATION | CCR/Weyl-cocycle and compact-packet stress integral giving D_R=2πdE; noncommuting first-law/Hessian and dephasing comparison; 2D curvature obstruction; conditional horizon coefficient/null-tensor solve; finite-switch one-particle response with independent convolution; one-particle energy and ΔK_R=2 |
| NEW CONDITIONAL INTERPRETATION | A single specified state/access structure organizes distinguishability, stress weighting and observer temperature. For this model separate phenomenological postulates of a thermal right state or a linear information–energy law are unnecessary because both follow from the same parent inputs |
| UNPROVED GRAVITATIONAL HYPOTHESIS | That geometric entropy/area balance and universal local equilibrium follow from this QFT, or that the metric adjusts to satisfy them. Neither follows; the standard higher-dimensional implication is conditional on G1–G6 |

**What was actually compressed?** Within the given QFT, one does not independently assume the thermal response, the coefficient 2π in the modular weight and the coherent distinguishability law. They are linked by the vacuum analytic/modular structure and the canonical action. This is a real conventional-QFT connection. No comparison proves fewer foundational axioms than quantum field theory itself. Merely calling the resulting package “OBP” would remove no further assumption.

**Essential assumptions:** reference state, accessible algebra and positive quantum probability structure; local relativistic dynamics for the geometric boost identification; a valid excitation/domain; fixed preparation and detector coupling for operational probabilities. General modular positivity needs no PT, quaternionic structure, horizon or gravitational action. Removing the vacuum/wedge structure generally removes the simple stress-integral flow. Removing area entropy or Clausius destroys the inference to a metric equation.

The calculation explains a quantitative physical relationship but does not equate relative entropy, thermal entropy and entropy production. D is a two-state/algebra comparison; KMS thermality is relative to a particular flow; accessibility limits which comparisons are measurable; entropy production requires a physical process and, for an arrow, additional asymmetric preparation/coarse-graining/boundary ingredients. Relative entropy is invariant under common algebra automorphisms, not a universal physical-time growth function.

## 5. Local project context, without retroactive claims

- I01 `01_PHYSICAL_PROBLEM_AND_AXIOMS.md` and `02_EXACT_OBSERVER_REDUCTION.md` fix the parent, IR-safe domain and detector. Its `03_INFORMATION_CAUSALITY_AND_TIME.md` and evidence supply the one-particle wave amplitude and type distinctions. Their files/tests are read-only historical evidence, not counted again as independent experiments.
- D02 `01_PARENT_MODEL_AND_DERIVATION.md`, `02_FIXED_PARENT_PROJECTION_TEST.md`, `03_PHYSICS_NOVELTY_AND_NEXT_DECISION.md` and actual derivation scripts were consulted. Its torsion response follows from its supplied EH/matter action; it is not an information-derived gravitational law in the different parent here.
- OBP `refs/preprints202604.1054.v3.pdf`, §§2–5.6, is an organizing/conditional framework; Probability `refs/preprints202604.0888.v2.pdf`, §§2–4, supplies conditional operational-equivalence criteria; Particles9,93, exact local `/Users/rocky/Downloads/particles-09-00093.pdf`, p.2 Eqs.(1)–(2), supplies a strict/Schur operator distinction. None supplies an area law, the Araki theorem, or this parent's backreaction. They are controls, not substituted proofs.
- The internal predictivity whitepaper, §§1,13–18, remains unchanged. No map from these results to (f1,Z_geom) was made, no functional rank reduction was shown, and **IB-2 / constitutive and empirical predictivity work remain CLOSED**. No historical Symmetry audit or manuscript was modified.

## 6. Strongest negative result and one next physical question

The strongest negative inference is precise: **a valid information–energy identity on a fixed geometry does not imply a law for that geometry.** The actual 2D parent has D_R>0 and stress response while G_ab≡0; its entanglement first law and modular positivity remain true. Area entropy plus a universal equilibrium coupling are additional physical content, not algebraic consequences. Conversely the conditional higher-dimensional calculation is a real mathematical bridge once those inputs are declared, rather than an independent derivation of them.

The single recommended next question is: **for I01's specified one-particle excitation, what is its actual wedge Araki relative entropy, beyond the already computable ΔK_R=2?** Calculating that non-Gaussian quantity would test how much of the information–energy relation survives beyond a local coherent unitary, without pretending its unknown entropy term is zero. It uses the same parent and a preparation already operationally distinguishable here. No such follow-on calculation, black-hole extension or new gravity model has been started.

## 7. Execution and stop

Final outputs: `modular_results.json` has 21 exact symbolic/algebraic checks and one compact-packet quadrature check; `geometry_results.json` has 9 exact checks; `detector_results.json` records two finite-duration responses and the unwindowed amplitude check; `independent_results.json` has 10 independently implemented test methods. Analytic proofs and assumed existing theorems are distinguished in the reports. Numbers verify formulas, not new empirical physics. Scripts import no project production implementation or prior result JSON.

`evidence/README.md` gives reproduction/source notes. `initial_state.json` records the pre-sprint baseline; `preservation_final.json` verifies original bytes and Git state; `deliverable_sha256.json` checks new deliverables. All new files are confined to `artifacts/discovery/PTQ_CLOUDS_I02/`. No Git mutation, dependency installation, fitting, publication or external communication occurred.

**NEWLY EXECUTED RESEARCH IS NOT RETROACTIVE PROOF OF THE ORIGINAL SYMMETRY ARTICLE.**

**STOP. Bounded sprint complete; no automatic continuation.**
