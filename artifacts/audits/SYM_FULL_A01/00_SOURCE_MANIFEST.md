# PTQ-SYM-FULL-CHAIN-AUDIT-01 — source manifest

Audit date: 2026-10-19 (Asia/Shanghai). Original working directory: `/Users/rocky/Downloads/palatini-pt-spurion`.
Audit output: `/Users/rocky/Downloads/palatini-pt-spurion/artifacts/audits/SYM_FULL_A01`. This directory did not previously exist. No earlier run was overwritten.

## Initial state, captured before source inspection

```
pwd: /Users/rocky/Downloads/palatini-pt-spurion
HEAD: 6b456b89baa866b6dd05bc69a837b98a985a34ad
branch: main
git status --porcelain=v1 --untracked-files=all:
?? symmetry-18-00170-v2.pdf
```

No tracked modifications were present. The local primary PDF was untracked; it is nevertheless preserved scientific evidence. `evidence/initial_state.json` was generated immediately after the first manifest file was written, so its status additionally contains that new audit manifest. The block above records the truly initial status from the first read-only command; this timing difference is explicit.

## Hash boundary and preservation

All **131 pre-existing regular files outside .git** were hashed with SHA-256 before code execution or PDF extraction. This includes source, configurations, tests, notebooks, all documentation, PDF/PNG/data figures, MD5 sidecars, build files, and the paper. Full path/digest records: `evidence/original_sha256.json`. `.git` internal metadata is not part of the scientific-file hash manifest; HEAD and branch are separately verified. The only new top-level content is the authorized audit subtree.

`working_copy/` is a byte copy of these sources, with `.git` and the audit output excluded. Project-code execution used the isolated source copy; final outputs, pytest files and matplotlib caches are inside the audit subtree. A newly created root pytest cache from the first independent development invocation was detected and relocated intact at final verification; 10_STATE and evidence/cache_relocation.json record this transient output-location mistake. Original code/configurations/figures/notebooks were never edited. No checkout/reset/clean/rebase/pull/commit/push/tag/branch changes, package installation, or external communications were performed. Read-only HTTP requests fetched publication metadata and a PDF; no downloaded code was executed.

Final preservation outcome is recorded in `10_STATE.md` and `evidence/preservation_final.json`.

## Exact primary PDF identity

- Title: *Guaranteed Tensor Luminality from Symmetry: A PT-Even Palatini Torsion Framework*.
- Author: Chien-Chih Chen. Journal: Symmetry 2026, 18, 170. DOI: `10.3390/sym18010170`.
- Local filename: `symmetry-18-00170-v2.pdf`; **50 pages**, **1,107,105 bytes**.
- SHA-256: `4f949b70c09b6e54d22c318d6c8a74be03002c9fb3852a6883aa0de8fcdc619d`.
- Publisher v2 PDF downloaded read-only from [MDPI publication artifact](https://mdpi-res.com/d_attachment/symmetry/symmetry-18-00170/article_deploy/symmetry-18-00170-v2.pdf) into `evidence/publisher_v2.pdf`: **byte-identical** to local primary source.
- DOI metadata: `evidence/crossref_metadata.json`, retrieved from [Crossref DOI record](https://api.crossref.org/works/10.3390/sym18010170).
- Paper page 1: received 20 December 2025; revised 9 January 2026; accepted 13 January 2026; published 16 January 2026. Embedded creation/modification dates are 21 January 2026. Those PDF processing dates do not imply a different scientific edition. No claim is made about an undocumented earlier v1 or later revision.
- PDF text: `evidence/paper_page_01.txt` through `paper_page_50.txt`, plus `paper_full.txt`. Selected pages 31,32,34,35 also have independently rendered page evidence. Equation transcriptions on pp.31–34 were checked visually, not inferred from filenames.

## Environment

Existing Python: `/Library/Developer/CommandLineTools/usr/bin/python3`, version `3.9.6`. Platform: `macOS-26.6.2-arm64-arm-64bit`.

**MISSING_ENVIRONMENT:** the project declares Python >=3.10, and environment.yml requests 3.11. The existing system interpreter with the required scientific libraries is 3.9.6. It can execute the existing 44 collected tests, but cannot execute one figure module's unpostponed `dict | None` annotations. Baseline runs below are explicitly compatibility-limited diagnostics, not certification of a supported environment. An existing Blender-bundled Python 3.11.11 has NumPy 1.26.4 but lacks SymPy, matplotlib, YAML, pytest and pandas. Two discovered virtual environments have broken interpreter links. No dependencies were installed or borrowed from another repository as authoritative scientific source.

Installed relevant package versions in the used environment:

|Package|Version|
|---|---|
|numpy|2.0.2|
|sympy|1.14.0|
|matplotlib|3.9.4|
|PyYAML|6.0.3|
|pytest|8.4.2|
|pandas|2.3.3|
|pypdf|6.14.2|

The complete installed distribution inventory is in `evidence/initial_state.json`. Requirements are lower bounds rather than an exact dependency lock; the historical smoke manifest records Python 3.12.3 and a different commit (`55f14b8`). This is a provenance mismatch, not proof that the currently audited source is corrupted.

## Evidence hierarchy

L0 published statement; L1 explicit derivation; L2 actual action-to-coefficient source derivation; L3 independent numerical reproduction; L4 configured/demo result; L5 figure regeneration; L6 documentation assertion. An L1 counterexample can defeat an L0 theorem even when its L4/L5 demonstrations reproduce. Integrity hashes certify bytes, not mathematical truth. Independent analytic tests below are L1 checks implemented with exact arithmetic; software-provenance probes remain software checks.
