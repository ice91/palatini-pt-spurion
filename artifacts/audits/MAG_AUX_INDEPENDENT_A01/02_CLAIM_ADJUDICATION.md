# Mathematical claim adjudication

Evidence: exact derivation in `01_INDEPENDENT_DERIVATION.md`, symbolic polynomial outputs and 40 independent checks in `evidence/`. PASS means established within the manuscript's explicit domain; it does not promote an auxiliary statement into a spacetime dynamics theorem. Confidence is high for explicit identities/counterexamples, and limited to source inspection for absent physical derivations.

## Requested ten checks

| ID | Mathematical claim and exact source | Verdict | Independent result and limitation |
|---|---|---|---|
| M01 | Stated trace-sector action and connection conventions, pp. 3–4, (3.1)–(3.7), Assumptions 3.1–3.3 | PASS | Trace map inverted directly from definitions; A included among varied fields. This is a defined trace/vector model, with excluded irreps outside the domain. Barred divergence must mean Levi-Civita, as used explicitly by (5.5)–(5.6). |
| M02 | Trace curvature expansion, p. 6 (5.1), Appendix C | PASS | Both derivative contraction and six quadratic coefficients independently recovered from curvature. No omitted curvature contribution within this domain. |
| M03 | Full auxiliary Hessian including A, pp. 7–10 (5.13), (6.2)–(6.12), Appendix F | PASS | Direct second differentiation of the action-derived bulk polynomial gives the raw 4×4 species matrix and full 16×16 Lorentz-paired Hessian. |
| M04 | Projective invariance for arbitrary xi, pp. 4–6 (4.1)–(4.15), Appendix A | PASS | Exact curvature shift for all 256 components and arbitrary xi jets; scalar curvature, residue, and currents invariant. No exact-form restriction and no metric-compatibility restriction. Selection of the law is within the adopted two-parameter family and specified targets. |
| M05 | Exact kernel and regular locus, pp. 9–11, (6.1)–(6.12) | PASS | det Hperp=−f1²f2/6; precisely f1f2≠0 gives rank12/nullity4 and kernel equal to specified projective image. All singular strata checked separately. |
| M06 | Schur elimination, stationary orbit and reduced action, pp. 8–13, (5.19)–(5.29), (6.13)–(6.15), (7.1)–(7.3) | PASS | Derived St=2f2, effective source=−df3, Palatini term=3(df1)²/(2f1), and residue term=−(df3)²/(4f2). Raw and quotient solves agree without supplying these target values as inputs. Literal intermediate (G.5) fails, as separately adjudicated below. |
| M07 | f1=0 and/or f2=0 branches, pp. 13–15 Table 2, Appendix I | PASS | Direct rank/nullspace/source computation without singular division gives S2: df1=0; S3: df3=0; S4: both. Compatible invariant moduli remain; no dynamical crossing result follows. |
| M08 | Quotient/elimination commutativity, Theorems 2.1, 6.3, 7.2; Assumption 4.4; Appendix H | CONDITIONAL | Proved as the explicitly stated local bulk equivalence modulo the two displayed currents on a smooth regular region. This does not establish equality of unrestricted boundary-value variational problems or projective charges. The source already states these conditions. |
| M09 | Any extension to spacetime dynamics, stability, propagating DoF, pp. 13, 15–17 | MISSING_SOURCE | No ADM reduction, physical Poisson/Dirac algebra, tensor principal symbol, or stability proof is supplied or needed for the auxiliary theorem. The manuscript explicitly excludes these claims; their absence is not a failed claim of two physical DoF. |
| M10 | Dependence on published Symmetry C1–C3, pp. 2, 16–17, references [14]–[16] | PASS | Present proof can be reconstructed from this action and conventions alone. No C1 eta, C2 density equivalence, C3 weights/mixing matrix/K=G or PT projector is an input. Historical prose is not an independent verification of earlier claims. |

## Other mathematical claims requiring separate verdicts

| Claim | Location | Verdict | Reason |
|---|---|---|---|
| Quadratic auxiliary reduction criterion | pp. 2–3, Theorem 2.1 | PASS | Symmetric H, HG=0, GᵀJ=0, and ker H=im G imply an invertible induced quotient form. Solutions constitute exactly one affine gauge orbit and the value L0+JᵀX/2 is well defined. This is finite-dimensional linear algebra. |
| Necessity of kernel equality for unique quotient solve for every compatible source | p. 3, after Remark 2.2 | PASS | An extra kernel vector is a distinct quotient solution whenever a solution exists; hence uniqueness fails. Necessity of equal on-shell values for a single source is correctly not claimed. |
| Species-to-full-fibre lift | p. 3, Remark 2.2 | PASS | Metric nondegeneracy gives ker(Hsp⊗g⁻¹)=ker Hsp⊗T* and rank multiplied by four. Indefinite Lorentz signature does not alter this conclusion. |
| Same regular locus under invertible horizontal charts | Theorem 2.1(5), Appendix B | PASS | Congruence multiplies species determinant by (det GL)²; the invariant on-shell scalar is unchanged. |
| Smooth local orbit of connection/A solutions | pp. 10–11, Theorem 6.2 | PASS | The explicit rational solution is smooth wherever both denominators remain nonzero; arbitrary V supplies the projective orbit. No global gauge slice is inferred. |
| Main four statements of Theorem 7.2 | p. 13 | PASS | They hold with its explicit trace/vector, metric-minimal matter, smooth regular-patch and admitted-boundary hypotheses. None requires the erroneous intermediate (G.5); direct substitution recovers the stated final result. |
| Invariance of actual IBP currents under projective shifts | Appendices G–H, pp. 23–24 | PASS | W and t are each invariant off shell with inert spectators. This is invariance under the specified transformation, not vanishing of a general variational boundary flux. |
| Literal intermediate equality (G.5) | p. 23, Appendix G.2 | CONTRADICTED | Its displayed source term retains 2s even though the coefficients of s cancel in 2T+Q−U. Residual relative to its RHS is (df1·df3)/f2. |
| Quadratic and source correction formulas (J.5), (J.6) | p. 26, Appendix J | PASS | Independently recomputing the Schur complement in the manuscript's diagnostic family gives the printed formulas. No deformation is made to the original action in this audit. |
| Simultaneous fingerprint zeros occur only at the undeformed point, without source qualification | p. 16 Proposition 9.1; p. 26 after (J.6) | CONTRADICTED | For allowed df1=0 the source correction vanishes identically, including nontrivial points on the quadratic conic. Counterexamples exist arbitrarily near the origin. The claim holds as an identity for arbitrary source jets, or where df1≠0; those quantifiers are missing from the unconditional wording. |
| Fingerprint corrections are generically nonzero in the printed two-parameter family | pp. 16, 26 | PASS | The zero set of the nonconstant quadratic Schur correction is a proper conic. Its being measure zero does not imply that it contains only one point. |
| Numerical inequality Keff≠Zgeom treated literally | p. 13, (7.6), Proposition 7.1 | MISSING_SOURCE | Zgeom has no mathematical definition here. Its intended prose meaning is that no identification is established. The actual Keff formula is derived; a universal inequality with an unspecified quantity is not a theorem. |
| Kinetic coefficient alone guarantees ghost freedom or a propagating scalar | pp. 13, 15–16 | MISSING_SOURCE | The source explicitly declines this extension. Whether phi is dynamical is conditional, and metric-scalar dynamics have not been analyzed by an auxiliary Hessian. |
| Deleted irreps vanish dynamically | Assumption 3.1; Table 3; p. 16 | MISSING_SOURCE | Not claimed: they are excluded coordinates. No equations for them were derived. |
| PT-specific physical mechanism or tensor luminality | no theorem in this manuscript | MISSING_SOURCE | The action-to-auxiliary chain does not contain such a derivation. This absence has no adverse implication for its properly limited algebraic theorem. |

## Exact counterexamples and their impact

### G.5: local algebraic error; main result survives

Using the action-derived stationary orbit, not the disputed equality,

\[
2T+Q-U=2(s+V)+4r-\tfrac83(s+V)-r+\tfrac23(s+V)=3r.
\]

The expression printed in (G.5) instead uses \(2s+4r-r\). Its LHS minus the claimed RHS is

\[
2\,df_1\cdot s=\frac{df_1\cdot df_3}{f_2}.
\]

For coefficient functions of a scalar this is \(f'_1f'_3(\partial\phi)^2/f_2\), not an identity zero. Already a single nonzero spatial component of both covectors gives a counterexample. All f1, f2 and sources remain symbolic in the test; no physical coefficient is tuned. The omitted terms are −8s/3+2s/3. Their cancellation is present in the raw source and in the independent calculation. Equations (G.4), (G.6), (6.15), and (7.1) are correct within scope despite this erroneous intermediate line.

### Appendix J / Proposition 9.1: source-degenerate counterexample

Use only the diagnostic family already printed in (J.2). This audit does not add it to (3.7). Its corrections are

\[
\delta S_t=-\frac{16\alpha+4\beta}{3}
-\frac{-4\alpha^2+16\alpha\beta+11\beta^2}{3f_1},\qquad
\delta J_{t,eff}=\frac{p}{f_1}(4\alpha+\beta).
\]

Constant nonzero f1, hence p=0, is permitted by the original posture. At \(\alpha=4f_1,\beta=0\) both corrections vanish at a nonzero diagnostic point. To remove any ambiguity about the word “nearby,” there is an exact curve

\[
\alpha=\frac{4f_1(4+m)}{4-16m-11m^2},\quad \beta=m\alpha,
\quad m\longrightarrow-4,\ m\ne-4.
\]

Both corrections vanish, and these nonzero points approach the origin. This disproves pointwise uniqueness in an allowed background, not the generic non-genericity claim or the undeformed action's fingerprint. Requiring a polynomial identity for arbitrary p would instead force beta=−4alpha and then delta St=−36alpha²/f1. That is a precise, narrower valid statement, not an assumption imported into the original theorem.

### Boundary scope: why a stronger theorem is unavailable

Take a flat non-null boundary and constant nonzero f3. Under an arbitrary variation with normal component delta t=1, delta(J3·n)=f3, although J3 is exactly invariant under projective transformations. Therefore projective current invariance cannot by itself remove boundary variations. The source's Assumption 4.4 correctly avoids claiming that stronger result. The original density and the reduced bulk density are equivalent only in the admitted bulk/IBP category, or with corresponding boundary data and terms treated consistently.
