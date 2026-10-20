# Independent adversarial validation

## Final result

**42 passed, 4 skipped / NOT_EXECUTABLE; exit0.** These are **46 audit cases**, independent of the original44 collected cases. Ordinary scientific-library deprecation warnings:14. Execution time about1.23s in the recorded environment. Test source: `tests/test_independent_audit.py`; exact command, cwd, environmental isolation, source SHA-256 and exit code: `evidence/independent_command.json`. Full stdout/stderr: `evidence/independent.stdout.txt`, `independent.stderr.txt`.

A successful counterexample is an audit PASS while the contradicted published assertion fails. A skipped unavailable physical derivation is neither a claim PASS nor an artificial input calculation. Pure symbolic cases do not import project code. The final group explicitly imports only the **working copy** to inspect software provenance/behavior and is not counted as mathematical derivation of the theory.

## Required categories

|Category|Independent invariant or counterexample|Result / scientific meaning|
|---|---|---|
|1. Exact projector/tensor identities|Involution projector idempotence/selection; odd X=Y=x defeats A2; conformal2-form Hodge scaling defeats A9|Verified valid elementary projector; reproduced two formal counterexamples|
|2. C1 irreducible decomposition|Pure-trace trace/axial/norm; A14 and A15 non-idempotence; A20 axial norm; Eq16 trace sign; single-vector no-go; explicit EH trace/axial/q contractions; irrep degeneracy; full trace eta variation; compensator integrability; metric compatibility|Surviving pure-trace algebra/no-go and full-action obstructions independently established|
|3. ROD determinant coefficients|Exact diagonal traceless4x4 determinant expanded symbolically|Coefficient−x²/12 and A_star=lambda²/8 verified under C1|
|4. CM determinant coefficients|Actual rank-one metric determinant for sigma=±1; expansion and comparison|Coefficient−x²/18, A_CM=lambda²/12; mismatch verified|
|5. CS+/Nieh–Yan pure trace|Component wedge contractions e_A∧T^A and T_A∧T^A|Both zero; nonzero additional CS+ source NOT_EXECUTABLE|
|6. Rank/nullity|General invertible homogeneous system; intended rational rank1 matrix; literal decimal exact determinant|Published implication false; chosen rational matrix alone has kernel(2,3); decimals tolerance distinguished|
|7. Independence of configured targets|Temporary sentinel JSON readback; public alpha ignored; strict flag accepts toy core; strict outer fallback does throw; synthetic degeneracy alias; paper NLO config not read by core API|Software-provenance checks only; original coefficient files never modified|
|8. Total divergence versus physical operator|Fixed-total-weight boundary equivalence; integral f'=0 but integral f'(h')²≠0; A40/A38 explicit flat-current mismatch; auxiliary-elimination chain rule|Invalid action-equivalence inference reproduced|
|9. Tensor speed from independently derived K,G|Direct EH ADM Christoffel/spatial Ricci and extrinsic-curvature second variation for plus polarization; flat curvature-square kernel; FRW scalar–TT contraction|Conditional EH luminal principal branch; no original full weighted K,G calculation available|
|10. DoF dependencies|Constraint-count arithmetic with/without first-class assumptions; programmed two-entry alias|Arithmetic is conditional; physical constraint algebra NOT_EXECUTABLE|

Additional NLO invariants: C²=GB+2Ricci²−2R²/3, off-shell TT Ricci² nonzero, flat massless higher-derivative root z=0, and derivative order of (nabla Sigma)² is6. These test published operator claims rather than fitting assigned data.

## Four explicitly blocked checks

1. `test_original_mixing_entries_from_ADM`: no evaluated original c0,cH,cY or complete second-variation source.
2. `test_nonzero_CSplus_coefficient_from_operator`: no explicit nonzero PT/Hodge density and lambda normalization.
3. `test_original_locked_tensor_speed_from_action`: no original physical weighted K(w),G(w); prescribed return functions excluded.
4. `test_original_constraint_algebra`: no physical Poisson/Dirac brackets with the fixed external spurion.

These are missing-source obstructions, not dependency-installation problems. They cannot be resolved by setting a new target coefficient or creating a model, and no such work was attempted.

## Independence controls and limitations

- Exact SymPy rational arithmetic is used for projectors, determinant expansions, matrix rank, currents and derivative identities. The literal JSON-decimal matrix and the intended rational matrix are deliberately separate cases.
- The tensor EH test explicitly constructs the3D spatial connection/Ricci scalar from gamma=diag(1+u q,1−u q,1), then expands to u² and performs the single justified compact-support IBP. It does not assume the source's K=G=1 return function.
- FRW scalar–TT vanishing is a rotation/transversality contraction test, not a guessed replacement mixing matrix.
- Tests with sentinels are only readback probes of copied software; they neither change original targets nor represent physical coefficients.
- Four unavailable source calculations are skipped with explicit NOT_EXECUTABLE reasons; synthetic inputs are not generated to turn them into passes.
- Used Python3.9 is outside the declared project support range, as recorded in00/02. The exact algebraic counterexamples do not rely on the failed runtime annotation path. No supported full paper-runtime certification is claimed.
- Initial development output is preserved as `independent_first_run.stdout.txt/.stderr.txt`. Six early audit-check failures were checker bookkeeping issues (structural symbolic equality vs algebraic equality, unintended Python float division, and matching epsilon conventions), not six new physical findings. They were corrected only in audit tests. The final run additionally includes a direct EH second variation and explicit FRW/config/auxiliary-elimination checks. No original source changed to make any test pass.

## Re-run without installation

Use the exact command/environment saved in `evidence/independent_command.json`, from the audit working copy. All pytest temporary files and matplotlib configuration live under audit/runtime. Original and independent suite outputs are intentionally separate. None of the new tests is added to the repository's original tests directory.
