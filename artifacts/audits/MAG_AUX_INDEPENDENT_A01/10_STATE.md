# Final audit state

Task: MAG_AUX_INDEPENDENT_A01. Status: adjudication complete; stop.

Source actually audited: `refs/manuscript.pdf`, 28 pages, matching requested title; the exact requested filename `manuscript(2).pdf` was unavailable. Source SHA256 and identity limits are in `00_SOURCE_MANIFEST.md`.

## Final decisions

- Core action-to-Hessian-to-reduced-action chain: **PASS within the expressly defined trace/vector, varied-A, local regular bulk domain**.
- Actual projective invariance for arbitrary one-form xi: **PASS**.
- All three singular classes and compatibility conditions: **PASS** as pointwise algebraic statements, without dividing by zero.
- Quotient/elimination: **PASS in the admitted local bulk category; CONDITIONAL for a boundary-value extension**.
- Literal Appendix G (G.5) intermediate equality: **CONTRADICTED**. Correct final reduced bulk action independently survives.
- Proposition 9.1 / Appendix J unqualified simultaneous-zero uniqueness: **CONTRADICTED** on allowed df1=0 backgrounds. Valid as an identity for arbitrary sources; generic non-genericity statement survives.
- Numerical inequality with undefined Zgeom: **MISSING_SOURCE**; the intended absence of an established identification should be stated in words.
- Full MAG consistency, stability, propagating DoF, or PT-specific luminality: **MISSING_SOURCE for these extensions; not claimed by the bounded auxiliary theorem**.
- Publication readiness of these supplied bytes: **CONDITIONAL**, with a narrow manuscript revision recommended. No new physics is required for the core result.

## Verification and preservation

Final independent validation: **40 passed, 0 failed**, including four successful counterexample/scope checks. The exact check list and execution history, including the corrected audit-harness error, are in `04_INDEPENDENT_CHECKS.md`. Final derivation, validation and preservation commands each exited 0; final derivation/validation stderr is empty.

All **1029 preexisting regular files** in the initial manifest remain SHA256-identical. No file is missing or unexpectedly created outside this audit directory. This includes the original manuscript and both earlier audits. HEAD and branch remain `6b456b89baa866b6dd05bc69a837b98a985a34ad` / `main`. See `evidence/preservation_final.json`.

No original file was edited. No dependencies installed, no remote action, no new model, no physical coefficients invented, no ADM/Dirac theorem added, and no original audit modified. Further research or publication action has not been launched.

## Deliverables

1. `00_SOURCE_MANIFEST.md`
2. `01_INDEPENDENT_DERIVATION.md`
3. `02_CLAIM_ADJUDICATION.md`
4. `03_SCOPE_AND_PUBLICATION_DECISION.md`
5. `04_INDEPENDENT_CHECKS.md`
6. `10_STATE.md`
7. `evidence/`: source extractions, inspected renders, independent scripts/results, execution logs and preservation manifests.

Recommended single next action: a bounded manuscript revision addressing the two mathematical wording/equation defects and the notation clarifications identified in `03_SCOPE_AND_PUBLICATION_DECISION.md`; then recheck only the affected statements. Original evidence remains untouched.
