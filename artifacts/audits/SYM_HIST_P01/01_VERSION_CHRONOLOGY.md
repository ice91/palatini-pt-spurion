# SYM_HIST_P01 — version chronology and source boundary

## Decision in brief

**C. Historical files confirm unresolved gaps.** The four available PDFs and accessible Git history recover the earlier *constraint-enforced* action and several missing provenance facts, but no complete historical action-to-ADM-to-K,G-to-constraints derivation. The chronology materially refines SYM_FULL_A01; it does not reverse its central negative/conditional verdict. Earlier constraints are not imported into the final paper as repairs.

## Authoritative files actually present

All four requested files were found directly in `/Users/rocky/Downloads/palatini-pt-spurion/`; no guessed path or replacement repository was used. All PDF-page references in these reports are **physical file pages, including the preprint cover**. Thus v2 file p.46 is printed45 of48; v3 file p.41 is printed40 of43. Symmetry file and printed page numbers coincide. v1 has a cover and44 total pages, with no consistent body pagination visible on the cited early pages.

|Version|Posted/published date on source|PDF pages|Exact local filename|Title|
|---|---|---:|---|---|
|v1|2025-09-29|44|`preprints202509.2421.v1.pdf`|Palatini–PT-Even Torsion: Uniqueness, Three-Chain Equivalence, and Exact Luminal Tensors|
|v2|2025-10-06|49|`preprints202509.2421.v2.pdf`|A Palatini PT-Even Torsion Framework: Conditional Uniqueness, Bulk Rank-One Equivalences, and Tensor Luminality via Coefficient Locking|
|v3|2025-10-30|44|`preprints202509.2421.v3.pdf`|Guaranteed Tensor Luminality from Symmetry: A PT-Even Palatini Torsion Framework|
|symmetry|2026-01-16|50|`symmetry-18-00170-v2.pdf`|Guaranteed Tensor Luminality from Symmetry: A PT-Even Palatini Torsion Framework|

Dates above are publication/header dates, not inferred from Git commit times. The three Preprints PDFs have embedded creation dates on2025-11-11 (v1 01:46:14Z; v2 04:25:40Z; v3 08:36:05Z). These later file-processing timestamps do not prove that the mathematical content was changed on that date. Local filenames, DOI/header identities and metadata were checked; no claim of byte identity to the bytes originally served on each September/October posting date is made. The supplied local PDFs themselves are the authoritative evidence for this task.

SHA-256 values:

- `preprints202509.2421.v1.pdf`: `8999eb1be4e97c5781dea1bc01cedca00c4262245d180d531cde18e9258ddb18` (1439548 bytes).
- `preprints202509.2421.v2.pdf`: `0ee6532e1a78f6ec404bf0f53bb57da755d9a66abfeea6545b2f52d53cfe4bd9` (1455705 bytes).
- `preprints202509.2421.v3.pdf`: `7f8ccc35cce4170b66815fe90ae12bcbc3f2b7a54a421485b2212eef0ff0e9f1` (1464887 bytes).
- `symmetry-18-00170-v2.pdf`: `4f949b70c09b6e54d22c318d6c8a74be03002c9fb3852a6883aa0de8fcdc619d` (1107105 bytes).

Metadata and text: `evidence/pdf_metadata.json`, `evidence/{v1,v2,v3,symmetry}/page_XX.txt` and `full.txt`. Seventeen selected pages were also rendered and visually checked for fraction factors, sign placement and equation labels; rendering commands are retained. The Symmetry PDF hash matches the previously audited publisher-identical PDF in SYM_FULL_A01.

## H01 — action and assumption chronology [ASSUMPTION_CHANGED]

|Stage|Actual source content|Effect|
|---|---|---|
|v1, file pp.5,9,11–13|A6 explicitly enforces trace lock; bulk Eq11 augmented by multiplier Eq17; vector Eq19 includes Lambda. Theorem2 explicitly says “with the trace lock”. Prose nevertheless already says it is not an extra assumption.|Constrained action is the mathematical input, despite stronger explanatory rhetoric.|
|v2, file pp.5,10,12–14|Same A6; bulk Eq19 plus multiplier Eq26; new Proposition4 argues collinearity; Theorem6 retains lock; §6.2/file p.18 explicitly imposes K=G as conditionL2.|No removal of the constraint and no independent physical K,G derivation.|
|v2, file pp.45–47|AppendixG first appears: new Stueckelberg action A61, finite algebraic equation A62, asserted exact solution A63, separate hard limit/current A64.|Additional action and limiting prescription are actually supplied, but the finite-mass exact-solution inference is false.|
|v3, file pp.5,7–10|A6 now calls trace alignment a dynamical result, “not an initial constraint”; bulk Eq8 still augmented by multiplier Eq14; vector Eq15 and deltaLambda constraint remain.|Narrative/assumption reclassification precedes actual removal of multiplier.|
|v3, file pp.40–42|AppendixG essentially repeats v2, renumbered97–103; the same finite-mass problem remains.|No recovered finite-mass proof.|
|Symmetry, pp.5,8,11–12|A6 becomes two-derivative truncation/fixed spurion. Main C1 proof drops multiplier and varies alpha1 I1+alpha3 I3, Eq20→21, defining eta=−alpha3/(4alpha1).|First of these four sources with an explicitly unaugmented main C1 proof and a coupling-derived eta formula. The full EH contribution is still not included there.|
|Symmetry, pp.20,44–45|Hamiltonian narrative still refers to trace-lock multiplier; AppendixG still has a separate exact constraint/hard-limit implementation.|Residual mismatch between main proof, DoF story and supplementary completion.|

This dates the transition only within the available versions. No inaccessible intermediate manuscript is inferred. See02 for the complete action/variation comparison.

## H02 — accessible Git chronology [CONFIGURED_DEMONSTRATION; VERSION_MISMATCH]

Read-only Git commands recover **72 reachable commits**, **130 historical paths**, **256 unique file blobs**, of which **244 decode as UTF-8**. `git fsck --full --no-reflogs --unreachable` exited0 with no unreachable objects reported. The clone is not shallow. Local refs are main/origin refs; no version-matching preprint tags or separate LaTeX branch were found. No fetch, checkout, reset, object write, ref update or remote mutation was used.

|Author date / commit|Evidence recovered|
|---|---|
|2025-09-15 `c10b3310cad3d5354908f2cc3b636e3394b99305`|Earliest field_eq implementation directly sets T_vec=alpha*d_eps; no connection variation.|
|2025-09-15 `662fda4ec60c3c220c25732f9b5bf4661599315b`|Early quadratic_action defines a chosen linear speed combination and positive normalization.|
|2025-09-18 `90495ef560638471c384a56e5999fb17cca8728b`|NLO parameterization introduced; no historical change to that module's sole blob.|
|2025-09-21 `1dfaf83279c378e1f8321661113d48e26af0c475`|Mixing config introduced as a grid/fit recipe with comments and symbolic:false; extraction script is an empty blob.|
|2025-09-21 `b24e203034c419b7a2cde2f8a8c4c823be58b669`|First nonempty mixing script stages JSON; says a compute path can be added later.|
|2025-09-21 `925736072ac3f9e89ca50c8dafa6b9c029ddb008`|Config replaced by explicit four numbers, including CM(divN)=+.35.|
|2025-09-21 `55f14b84668d9397ac0efb3578b3c4676e6ce96a`|CM(divN) edited to−.13333333333333333; ratio-checking/meta logic added in same commit. No ADM calculation supplied.|
|2025-09-22 `84501317edf5003e72cf54818075e4319bf9dcf9`|Last available2025 commit; earlier than all three posted preprint dates.|
|2026-01-23 to2026-01-29, ending `6b456b89baa866b6dd05bc69a837b98a985a34ad`|Citation/docs updates; no recovered original action-variation pipeline.|

Commit timestamps are recorded provenance, not independently certified execution dates. The same September code cannot be declared a convention-matched artifact for every later PDF without a version map. Relevant source files' historical alternatives were inspected, including all7 quadratic_action blobs,3 field_eq blobs,5 coeff_extractor blobs,4 degeneracy blobs and all mixing-script/config blobs.

Only two historical LaTeX-related assets exist: `tex/tikz/figA_basis_flow.tikz` and `tex/tikz/figB_conn_decomp.tikz`, both diagrams. **No manuscript .tex, .sty/.cls or mathematical derivation LaTeX archive is available in any recovered tree.** No claim is made that such a private archive never existed.

Full command stdout/stderr/exit codes: `evidence/git_commands.json`. Complete tree/blob inventories, read-only blob snapshots, chronological source patches and notebook cells are in evidence. No old code or notebook was executed. Nine unique historical notebook blobs contain no saved outputs and null execution counts; this does not prove they were never executed elsewhere, but no execution evidence is recoverable here.

## Preservation controls

Initial cwd/HEAD/branch/runtime and SHA-256 of **every pre-existing regular file outside .git**, including the entire existing SYM_FULL_A01 audit, were recorded before extraction. Existing files and prior audits were not edited. New reports/evidence live only in `artifacts/audits/SYM_HIST_P01/`. No dependencies were installed; the limited equation checks use existing Python3.9.6/SymPy1.14.0 and do not execute project code. Final preservation verification is in `evidence/preservation_final.json` and06_RESEARCH_DECISION.md. Git internals are observed read-only through Git commands; scientific-file hashes do not purport to hash live .git metadata.
