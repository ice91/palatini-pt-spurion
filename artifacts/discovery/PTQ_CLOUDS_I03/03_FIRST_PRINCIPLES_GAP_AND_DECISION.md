# I03 — microscopic matching and the remaining first-principles gap

**Decision:** the same scalar degrees of freedom generate both a planar matter-entanglement area term and an EH-like local curvature term in the background effective action. Their coefficients match precisely in the specified minimal/common-regulator calculation. This explains a real known connection; it does not determine a finite Newton constant, promote the metric to a physical dynamical field, or derive universal local Clausius equilibrium.

## 1. Completed calculations, not a source-only audit

One fixed free real scalar in 3+1 dimensions was used, with m>0, ξ=0, canonical Gaussian measure and proper-time lower cutoff ε². The metric was a source. There were two main calculations and no physical model revision:

1. **Cone spectrum → matter entropy.** Angular/radial scalar modes, a bulk-subtracted Abel trace and a transverse Gaussian integral give C(n)=(n⁻¹−n)/12 and S_ent/A=J/(48π). Neither G nor an entropy-area coefficient was supplied.
2. **Smooth heat kernel → curvature coefficient.** Heat-equation transport gives a_1=R/6; the same regulated determinant gives c_R=−J/(192π²). A separate S² spectrum and smooth-cap curvature integral check its geometry and normalization.

Here J=∫_{ε²}^∞ds e^{−m²s}/s², evaluated in report 01. The resulting relation is

\[
 \frac{S_{\rm ent}^{\epsilon}}A
 =-4\pi c_R^\epsilon=\frac14\delta(1/G),
 \qquad \delta(1/G)=\frac{J}{12\pi}.                         \tag{33}
\]

The relation is a newly executed benchmark of established physics. Susskind–Uglum, *PRD*50,2700 (1994), [primary abstract](https://doi.org/10.1103/PhysRevD.50.2700), established the historical entropy/coupling-renormalization connection. Our calculation is the specified planar minimal scalar calculation, not a reconstruction of their string-theory or black-hole claims.

## 2. Six decisive questions

| Question | Decision | Domain / missing premise |
|---|---|---|
| Q1 Does the matter theory produce an area entropy term? | **YES, regulated** | Vacuum, planar cut, specified scalar domain; entropy density diverges as ε→0 |
| Q2 Does the same field induce an EH-like term? | **YES** | Local weak-curvature term in a background generating functional; no metric propagation implied |
| Q3 Do the coefficients obey a precise relation? | **YES, (33)** | Minimal coupling, same covariant n-independent regulator and conical derivative at n=1; absolute coefficients are scheme dependent |
| Q4 Is a finite physical Newton constant determined without independent bare data? | **NO** | 1/G_R=1/G_B+J/(12π); arbitrary finite normalization remains. Setting the bare term to zero is extra data, and ε is not a derived physical scale |
| Q5 Is a consistent dynamical metric independently derived? | **NO** | Requires a physical metric variable, its dynamical principle, renormalization/volume/higher-curvature data and causal state/boundary prescription; none follows from a source determinant |
| Q6 Is I02's universal local equilibrium condition explained? | **NO** | Replica gluing and vacuum restriction do not imply δQ=TδS_h for every physical local horizon or derive its backreaction |

The YES answers are not promoted into the NO answers. There is no no-go theorem against every possible induced-gravity UV completion; this particular continuum matter calculation leaves independent choices demonstrably unfixed.

## 3. Assumption removal relative to I02 G3/G5

I02 `02_ENERGY_GEOMETRY_INTERFACE.md` separated G3 (geometric entropy/area with coefficient α) from G5 (universal equilibrium heat/area balance). I03 improves the microscopic bookkeeping of **part** of G3:

- The scalar vacuum area coefficient is computed, rather than chosen.
- The scalar's contribution to the curvature coefficient is computed independently and has the matching conical entropy.
- One need not fit two unrelated parameters for those two scalar-loop contributions.

What has not been removed: the finite total α=1/(4G_R); an identification of complete physical horizon entropy, including the gravitational sector; a physical UV completion; or the assertion that metric/horizon variations obey the Clausius condition. A bare gravitational entropy is not counted by the scalar Hilbert space merely because its EH action has a replica derivative. A coefficient matching in a source functional is weaker than universal state-dependent horizon dynamics.

The replica parameter n is a definition of a state functional through Euclidean gluing. It is not an affine physical time, an entropy-producing evolution, or the variation of area caused by energy flux. In particular no stationarity/entropy balance principle has been imposed on physical horizons by the calculation. Vacuum KMS behavior from I01 remains compatible with zero entropy production.

## 4. Adversarial controls and explicit failures of stronger claims

| Strong inference tested | Calculation / counterexample | Outcome |
|---|---|---|
| The area coefficient is independent of regulator | Smooth cubic proper-time profile gives J∼Γ(2/3)/ε² instead of ε⁻² | False as a bare absolute coefficient; common-profile matching survives |
| Matching removes species dependence | For N independent identical scalars, both S and δ(1/G) multiply by N | False; a consistent relation is not a prediction independent of spectrum/UV scale |
| Matter fixes the total curvature coefficient | A finite background term b∫R cancels from all normalized matter correlators but changes total c_R | False; report 02 Eq.(31) is a direct functional counterexample |
| Zero bare curvature coefficient is derived | Any desired finite 1/G_R can be obtained by Eq.(30) of report 02 | False; zero bare coupling is a UV normalization hypothesis |
| Correct EH sign guarantees a consistent flat metric solution | The same loop contains a quartically sensitive volume term; flat stationarity would require its independent cancellation | False; a nondynamical background itself is not inconsistent |
| EH matching reconstructs all entanglement information | Exact leading Rényi ratio S_2/S_1=3/4; EH-only finite-angle extrapolation gives 1 | False; the n→1 derivative is not the full replica functional |
| The metric/stress must encode all quantum information | Sign-related coherent preparations have the same quadratic stress but opposite calibrated local current means | False; state distinctions are not exhausted by stress expectation |
| The determinant already supplies gravitational equations | Covariance gives a stress Ward identity, not δΓ/δg=0 | False without an independent metric-dynamics postulate |

The coherent-state control is the I02 logical control, not a new independent experiment here. It applies to the same real scalar algebra in four dimensions: for compact classical displacement data, 〈:T_μν:〉 is quadratic and invariant under reversing the displacement, while a fixed smeared linear field/current has opposite means. A common field/observable redefinition can relate descriptions, but with the calibrated observable held fixed these are different preparations. No new universe sectors are inferred.

### Nonminimal coupling: scope warning, not an added model

The parent was not changed to ξ≠0 and no second nonminimal replica calculation was executed. The known boundary of the result matters: with a hypothetical ξRφ² term the smooth local coefficient would be (1/6−ξ)R, while the flat-space field operator/vacuum matter entanglement is unchanged at R=0. A conical continuation then requires additional tip/contact or Wald-like entropy bookkeeping; its geometric entropy must not be silently identified with the same flat matter entropy. At conformal ξ=1/6 the leading induced R coefficient vanishes although flat scalar matter entanglement does not. This falsifies a naive all-couplings extrapolation of the minimal result, not the computed ξ=0 identity.

These are established scope facts, not a repaired parent: Solodukhin, §3.16, Eqs.(141)–(144), [author text](https://arxiv.org/html/1104.3712). Domain choices for a singular nonminimal operator would have to be specified before claiming its physical entropy. No ξ was tuned to create a desired outcome in this sprint.

### Missing gravitational edge/constraint content

An ordinary real scalar has no gauge Gauss-law constraint requiring gauge-edge modes in this calculation. The background metric is not quantized. Consequently no graviton modes, diffeomorphism constraints, gravitational edge algebra or its state/measure are counted. It would be incorrect to say that their entropy has been shown to vanish, or that the scalar count proves the complete entropy of dynamical gravity. Gauge, nonminimal and graviton sectors require their own contact/constraint analysis and are outside the one-field result.

## 5. Distinguish the mathematical operations

| Operation | Actual object / role | Not supplied by it |
|---|---|---|
| State restriction | ω→ω|_A_R, inaccessible observables removed from reporting | A new metric action or fundamental global information erasure |
| Effective-action integration | ∫Dφ e^{−S_E[φ;g]} produces det D_g and a functional of external g | A postulate to integrate or vary g as a physical field |
| Hilbert projection | A typed idempotent P on state space; selective projection needs normalization/probability data | The replica trace, a determinant or a thermal state by itself |
| Schur elimination | A−BF⁻¹C for a declared block operator and inverse | The whole one-loop determinant; integrating a Gaussian block also gives a determinant factor |
| Geometric entropy | Conical derivative of a specified action functional, including any geometric/contact terms | Automatic equality to matter entropy for every coupling or a full microscopic gravitational state count |
| Dynamical field equation | Stationarity/path integration for variables postulated physical and dynamical | A consequence of merely writing a covariant source functional |

For the minimal scalar vacuum with this regulator, replica state gluing and conical differentiation give the matter entropy in report 01. This specific correspondence does not identify all the operations in the table. OBP can organize the package of state, accessible algebra and effective description, but supplies no missing determinant coefficient, UV scale or metric variational law. Probability and Particles results are not invoked as proofs of heat-kernel entropy or gravitational dynamics; their positive-probability and strict/Schur distinctions remain respected.

## 6. What this teaches about an information origin of geometry

There is an actual mathematical connection: the same short-distance matter fluctuations give a surface contribution under restricting/gluing the vacuum and a bulk curvature response under varying the source geometry. The transverse mode count explains area scaling; the scalar heat-kernel coefficient and conical curvature normalization explain the shared coupling coefficient. The loop-induced geometric entropy is therefore not an unrelated fitted number.

This is explanatory compression **within a declared QFT/regulator sector**, not a reduction of all gravitational assumptions. The total renormalized coupling still contains independent source normalization. A metric that responds dynamically to matter is an additional physical hypothesis, not a theorem about coarse graining. PT, quaternionic geometry and naming this architecture PTQ can all be removed without changing the calculation.

The most important remaining independent premise for a law of gravity is: **the metric itself is a physical dynamical variable whose evolution is governed by a complete, consistently renormalized action/state prescription.** Its UV completion must also fix the independent gravitational normalization data if one seeks a numerical prediction of G. This sprint supplies neither premise by quantum-information terminology.

## 7. One next calculation, not begun

The single recommended calculation is the **renormalized real-time retarded stress-tensor response kernel of this same scalar to a weak metric perturbation**, with its independent local counterterms left explicit. It would separate causal nonlocal response from scheme-dependent local contact terms, and test whether an explicitly postulated low-energy metric dynamics could have a self-consistent causal/stable regime. That is a concrete step toward the missing dynamics question; it would still not fix bare gravity by itself or prove universal equilibrium. No such calculation, new field model or follow-on sprint has started.

## 8. Epistemic classification, preservation and stop

- **ESTABLISHED PHYSICS RECONSTRUCTED:** scalar replica area entropy, local induced-curvature renormalization and their minimal common-regulator relation.
- **NEWLY EXECUTED CALCULATION:** the cone angular/radial sum, normalized replica derivative/Rényi control, RNC transport contraction, massive proper-time integrals, independent sphere/image/cap tests, and explicit regulator/counterterm controls.
- **CONDITIONAL INTERPRETATION:** this part of area entropy and induced coupling has a common microscopic accounting. An all-induced cutoff gravity interpretation would need further UV and metric-dynamics assumptions.
- **UNPROVED PHYSICAL HYPOTHESES:** complete horizon entropy, a finite predicted G, dynamical healthy gravity, universal local Clausius equilibrium, black-hole information recovery or quantum gravity.

Executed evidence: 14 exact checks in `area_results.json`, 18 in `curvature_results.json`, and 11 independently implemented test methods in `independent_results.json`. Numerical integrations/spectral extrapolations are verification illustrations, not empirical fits. One initial symbolic half-angle normalization failure is preserved in `area_attempt1.json`; its correction changed neither model nor expected identity. No failed physical candidate was hidden or tuned away.

All new files are in `artifacts/discovery/PTQ_CLOUDS_I03/`. `evidence/initial_state.json` records initial HEAD, main branch, status, refs, untracked inventory and hashes of 1176 pre-existing regular project files. `preservation_final.json` verifies their preservation and write scope; `deliverable_sha256.json` verifies the final new files. See `evidence/README.md` for reproduction and source notes.

Previous I01/I02 reports, historical Symmetry audits, whitepapers, source and manuscripts remain read-only evidence. **IB-2, empirical fitting and the constitutive-predictivity freeze remain CLOSED.** No Git mutation, installation, publication or external communication occurred. This work is not retroactive proof of Symmetry and does not solve black-hole information or quantum gravity.

**STOP after this bounded calculation. No automatic continuation.**
