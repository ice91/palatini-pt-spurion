# Root-cause analysis

The fundamental gap is **missing action-to-coefficient provenance combined with invalid mathematical steps**. Several programmed targets are reproducible, but the manuscript promotes their agreement into a physical derivation. Fixing a single printed determinant statement would not reconnect the full chain. This is an analysis of supplied evidence, not a publication-correction proposal.

## Classification key

A publication transcription error; B invalid mathematical inference; C physical coefficient provenance missing; D demonstration/hardcoded result treated as physical derivation; E non-equivalent action representations; F insufficient physical assumptions; G valid conditional result in narrower domain; H source/version mismatch. A is assigned only as a possibility when intent cannot be established; the audit does not excuse a repeated invalid proof by presuming a typo.

## RC1 — geometric symmetry does not act on the declared field space (B,F)

- Tried to prove: arbitrary affine projective one-form redundancy compensated by a fixed scalar gradient in a metric-compatible Palatini action.
- Actually computed: a formal trace shift and the symbol T−d epsilon, without a scalar transformation that realizes arbitrary xi or a metric-compatible transformed connection.
- First gap: Eq4→5, pp.4–5. d(delta epsilon)=3xi requires dxi=0; affine projective shift generates nonmetricity.
- Could conclusion survive? Some restricted geometry/symmetry could be consistent, but that is not the original declared posture and is not introduced here. The ordinary EH principal branch does not need this claimed gauge symmetry.
- Minimum evidence: lawful transformation on the originally declared variables, closed variational domain and full action invariance for exactly the claimed arbitrary transformations. A scalar exact-form identity cannot supply it as written.

## RC2 — irrep formulas and projector calculus inconsistent (B; possible A for individual signs)

- Tried: idempotent covariant projectors, consistent torsion norm, projection/variation calculus.
- Actually computed: pure-trace-only source examples; reconstruction residual defined to vanish; paper A14/A15 with opposite trace/axial extraction signs; A20 incorrect axial norm. A2 treats projection as if multiplicative. A9 fails a conformal Hodge test.
- First gaps: A2 p.28 and A14–A21 pp.31–32; Eq16 has the same trace sign issue relative to Eq10.
- Could conclusion survive? Elementary involution selection, restricted no-go and pure-trace IT contraction survive. General projector correctness does not.
- Minimum evidence: one consistent set of tensor definitions verified on all24 components and valid functional identities. The original axial software must be assessed beyond axial-zero inputs. The audit records contradictions without editing those formulas.

## RC3 — full curvature contribution omitted from connection variation (B,F,C)

- Tried: unique pure-trace dynamics of full Eq13 for alpha1≠0 and eta=−alpha3/(4alpha1).
- Actually computed: variation of the selected explicit torsion operators Eq20 only; software instead directly sets T=alpha*d epsilon.
- First gap: §4.1.2 transition from Eq13 to Eq20. R(e,omega) is also torsion-dependent; algebraic field-map invertibility is not invertibility of action Hessian blocks.
- Could conclusion survive? Yes, a pure-trace branch can arise on the nondegenerate actual-action domain, with a full-action coefficient relation. This is conditional and does not prove the printed eta/uniqueness/positivity statements.
- Minimum evidence: original complete connection variation including curvature, evaluated irrep coefficients, stated singular loci and allowed coefficient domain. No redefinition of alpha1 is assumed by the audit.

## RC4 — CM and ROD actions are not the same determinant (E,B)

- Tried: equality of two original volume-deformation routes with common A_star=lambda²/8.
- Actually computed: ROD uses traceless P−I/4; CM uses n n. Appendix C asserts their Jacobians equal, while code merely uses two equal JSON values.
- First gap: p.34 metric shift→displayed determinant ratio, before A34. Exact coefficients−x²/12 vs−x²/18 contradict equality.
- Could conclusion survive? ROD coefficient alone survives. The named original actions are not proved equivalent; no supplied boundary or field-redefinition calculation repairs the mismatch.
- Minimum evidence: a source-grounded equality of the actual declared action variations, or an explicit already-published legitimate transformation. Inventing subtraction/normalization is prohibited and was not done.

## RC5 — nonzero CS+ bulk asserted without an operator (C,E,B,D)

- Tried: third independent route with the same nonzero torsion invariant coefficient.
- Actually computed: ordinary NY identity, then an unsupported nonzero A35; source reads .125. Pure trace gives e∧T=T∧T=NY=0.
- First gap: the identity→A35 on p.34; no explicit extra PT/Hodge density or lambda normalization.
- Could conclusion survive? Ordinary NY remains an exact boundary identity. A distinct shadow operator might require independent definition, but none may be invented to rescue the claim.
- Minimum evidence: the exact original nonzero operator, its coefficient and full calculation, distinguishing T∧T from T∧star T. If no such original source exists the present nonzero route remains unsupported/contradicted in its ordinary reading.

## RC6 — invertible homogeneous system said to have a nonzero ray (B; A unconfirmed)

- Tried: unique nontrivial weight ratio on generic backgrounds.
- Actually computed: manuscript asserts noncollinearity/det≠0; repository stages nearly rank1 configured rows and checks equal ratios.
- First gap: Eq33→unique one-dimensional solution p.17. Nonzero determinant means only zero solution. Eq34 needs the opposite rank condition.
- Could conclusion survive? Rank1 matrices can have one-dimensional kernels, as the configured intended example does. Physical matrix rank and weight uniqueness still lack provenance.
- Minimum evidence: original evaluated ADM matrix and a mathematically valid rank/domain argument. Replacing≠0 by=0 in prose alone is insufficient. Repetition of noncollinearity in Appendix D prevents independently identifying this as a single typographical incident.

## RC7 — symbolic placeholders and staged numbers replace physical extraction (C,D,H)

- Tried: independently auditable ADM mixing tables and full action-to-GW pipeline.
- Actually computed: unnamed c coefficients in TablesA1/A2; four chosen JSON values staged by a script explicitly disclaiming physical extraction; speed/eigenvalue/waveform functions use assigned effective formulas.
- First gap: action→mixing coefficients and action→K,G. Metadata/MD5 do not add derivations.
- Could conclusion survive? A physical calculation might exist outside supplied evidence, but this audit cannot presume it. Programmed2:3 and cT=1 remain demonstrations.
- Minimum evidence: archived original symbolic second-variation expressions, evaluated entries and K,G tied to the same action/background/conventions. Another repository is not substituted as authoritative source.

## RC8 — boundary equivalence conflated with physical locking and equal kernels (B,E,F)

- Tried: bulk-equivalent routes select different TT-mixing weights; divergence of K−G forces luminality.
- Actually computed: identical canonical currents declared; flux curve set to approach1; background coefficients equated to h²-current divergence.
- First gaps: §5.3/p.16 claims boundary representatives change physical bulk mixing despite A4; Eq35/p.17 drops a divergence coefficient without tracking its perturbation factors.
- Could conclusion survive? A genuine full-density identity could prove equal principal symbols, but is absent. Ordinary EH equality survives separately; no unique route weight follows from irrelevant boundary differences at fixed total weight.
- Minimum evidence: complete reduced quadratic density, legitimate boundary variations and constraint elimination leading to principal symbols. A coefficient integral or a flux ratio is insufficient.

## RC9 — physical constraint algebra replaced by chosen eigenvalue count (C,D,F)

- Tried: exactly two physical DoF with GR first-class algebra in the full posture.
- Actually computed: conditional arithmetic12−8=4 phase dimensions; two synthetic numbers or two1s and38 zeros. A multiplier appears without an Eq13 multiplier action.
- First gap: Section7.2 assertion of first-class constraint closure; no bracket/preservation evidence with fixed external profile.
- Could conclusion survive? The TT sector of ordinary EH has two polarizations. Full metric-plus-fixed-spurion constraint content remains conditional; no global extra-mode conclusion is asserted by this audit.
- Minimum evidence: physical constrained Hamiltonian, constraint classes, closure/rank and background consistency on the original domain.

## RC10 — covariant higher-derivative operators treated as spatial-only dispersion (B,C,D,F)

- Tried: complete four-derivative basis, only Weyl² changes luminality, distinctive nonzero slope2 prediction.
- Actually computed: labels plus chosen DeltaK=a k²,DeltaG=(a+b)k²; figure directly plots b k²/Lambda². Table curvature directions are dependent; Ricci² has off-shell TT(Box h)²; Weyl² retains time-derivative partners and a luminal flat massless root.
- First gaps: Table2 p.9 “no TT”/unique-direction classification; App.H A81/A89/A91 omit required complete reduction/branch specification. `(nabla Sigma)²` is six derivative in the stated counting.
- Could conclusion survive? A k² ansatz has slope2 in its valid leading regime. A physical prediction from the claimed unique operator is not established; particular background-dependent corrections would need their actual original calculation.
- Minimum evidence: consistent basis quotient, derivative counting/normalization, full off-shell TT operator including time derivatives, and stated low-energy branch/order-reduction map to b. No new operators were added to address this.

## RC11 — reproducibility provenance and environment (H; not a physical claim failure)

- Tried: historical44 passed, pinned environment, strict APIs, manifest-tracked publication run.
- Actually computed now:42 passed+2 CLI skips; direct make paper-test lacks pytest command; module equivalent succeeds; partial make paper stops on Python3.9 annotation. All20 sidecars match. Manifest still references a2025 smoke run at different commit/Python.
- First gaps: dependency declarations are ranges, not a lock; strict flag tests API presence not physical derivation; make paper does not refresh manifest.
- Could conclusion survive? Supported-environment software reproduction may succeed. It would still not repair the analytic counterexamples or coefficient provenance.
- Minimum evidence: supported archived runtime and exact run manifest tied to source/data. No installation was authorized or attempted.

## RC12 — what is genuinely established (G)

The independently surviving mechanism is ordinary **algebraic auxiliary-field elimination plus the EH metric principal operator**, on a nonsingular, consistent domain. Linear single-vector covariance restricts sourced torsion irreps; pure trace gives IT=−6eta²Sigma. None of these narrower results shows that the claimed PT/projective symmetries uniquely select a physical weight ratio or make a distinctive tensor-speed prediction. Conditional generic luminality and PT-specific physical novelty must remain separate.

The audit makes no judgment about all possible PTQ models, the author's intent, or scientific misconduct. It addresses only the identified published statements and supplied code snapshot, with proof failures, missing sources and software-environment failures kept separate.
