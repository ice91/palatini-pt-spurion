# MAG_AUX_INDEPENDENT_A01 — source and preservation record

Audit date: 2026-10-19. Role: independent metric-affine mathematical auditor.

## Identified source

- Workspace: `/Users/rocky/Downloads/palatini-pt-spurion`.
- Requested filename: `manuscript(2).pdf`; that exact filename was not found.
- Actual source: `refs/manuscript.pdf`, 420099 bytes, 28 pages, matching the requested title **Projective Gauge Reduction and Auxiliary Connection Elimination in a Metric-Affine Trace Sector**. The user clarified that the preprints are in `refs`.
- SHA256: `78cf9ad3437e0e82c0ae12aaecf7c452d0946a7d71846b42a2a69984b818a4e8`.
- PDF creation/modification metadata: `D:20260924162715+08'00'`; producer `pdfTeX-1.40.29`. Metadata is identification evidence, not proof of publication history.
- All adjudications concern these exact bytes. Identity with an unavailable separately named `manuscript(2).pdf` is not asserted. No version label beyond the available PDF is invented.
- PDF physical and printed page numbers coincide. Equation references below refer to this PDF, not to the Symmetry paper.

## Preservation and execution

HEAD recorded: `6b456b89baa866b6dd05bc69a837b98a985a34ad`; branch `main`. The initial SHA256 manifest contains 1029 existing regular files outside `.git`, including previous audits and all available PDFs. Existing untracked material is evidence, not a cleanup target.

Only this fresh audit directory is written. No original Python module, test, notebook, PDF, configuration, figure, Git branch, index, or earlier audit is changed. No network request, dependency installation, original test execution, model repair, or external publication action was performed. The new action is audited directly; old software tests would not establish its mathematics.

The original repository is not executed. New scripts execute only their own local symbolic derivation using the already available interpreter, with `PYTHONDONTWRITEBYTECODE=1`. Their generated outputs remain in `evidence/`. This is sufficient isolation for the newly written audit code; no writable duplicate of original scientific code is needed.

Environment: Python 3.9.6, `/Library/Developer/CommandLineTools/usr/bin/python3`; SymPy 1.14.0; pypdf 6.14.2; Pillow 11.3.0. Exact runtime metadata is in `evidence/environment.json`.

## Evidence files

- `evidence/source_state.json`: source identity, metadata, HEAD, branch, initial file count.
- `evidence/initial_sha256.json`: preexisting-file preservation baseline.
- `evidence/page_01.txt` through `page_28.txt`, `manuscript_full.txt`: extracted text. Rotated diagrams are not used to infer equations.
- `evidence/visual_page_23_white.png`, `visual_page_26_white.png`: rendered pages independently inspected to confirm literal (G.5) and Appendix J. Transparent initial PNGs were composited onto white; source PDF untouched.
- `evidence/derive_from_action.py`: derivation from connection definitions and action; no proposed Hessian/solution/reduced coefficient is supplied as an input.
- `evidence/independent_derivation.json`: actual resulting contractions, Hessians, solves, and reduced polynomial.
- `evidence/validate_independently.py`, `independent_validation.json`: separate comparison/invariant/counterexample checks.
- `evidence/derive.stdout.txt`, `derive.stderr.txt`, `validation.stdout.txt`, `validation.stderr.txt`: execution logs.
- `evidence/validation_attempt1.*`: preserved failed audit-harness run. A Python name collision between the projective distortion function and Schur scalar caused a TypeError; it was corrected only in the audit script.
- `evidence/validation_attempt2.*`: successful 39-check run before adding an exact nearby-counterexample check. Final run has 40 checks.
- `evidence/git_status_after.txt`: full read-only Git status, including preexisting untracked files.
- `evidence/preservation_final.json`: final hashes, unexpected-file check, HEAD and branch comparison.

## Reproduction commands

Run from the workspace, without installations:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 artifacts/audits/MAG_AUX_INDEPENDENT_A01/evidence/derive_from_action.py
PYTHONDONTWRITEBYTECODE=1 python3 artifacts/audits/MAG_AUX_INDEPENDENT_A01/evidence/validate_independently.py
```

The scripts resolve their output location relative to their own file. Re-running replaces generated files in this audit directory only. Command exit codes and final preservation status are recorded in `10_STATE.md`.
