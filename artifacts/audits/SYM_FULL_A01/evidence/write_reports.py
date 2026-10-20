from pathlib import Path
import json,sys,hashlib
O=Path(__file__).resolve().parents[1]; E=O/'evidence'
def write(name,text):(O/name).write_text(text.strip()+'\n')
state=json.loads((E/'initial_state.json').read_text()); hashes=json.loads((E/'original_sha256.json').read_text())
write('00_SOURCE_MANIFEST.md',f'''
# PTQ-SYM-FULL-CHAIN-AUDIT-01 — source manifest

Audit date: 2026-10-19 (Asia/Shanghai). Original working directory: `{state['cwd']}`.
Audit output: `{O}`. This directory did not previously exist. No earlier run was overwritten.

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

All **{len(hashes)} pre-existing regular files outside .git** were hashed with SHA-256 before code execution or PDF extraction. This includes source, configurations, tests, notebooks, all documentation, PDF/PNG/data figures, MD5 sidecars, build files, and the paper. Full path/digest records: `evidence/original_sha256.json`. `.git` internal metadata is not part of the scientific-file hash manifest; HEAD and branch are separately verified. The only new top-level content is the authorized audit subtree.

`working_copy/` is a byte copy of these sources, with `.git` and the audit output excluded. All project execution, output generation, temporary pytest files and matplotlib caches are inside the audit subtree. Original code/configurations/figures/notebooks were never edited. No checkout/reset/clean/rebase/pull/commit/push/tag/branch changes, package installation, or external communications were performed. Read-only HTTP requests fetched publication metadata and a PDF; no downloaded code was executed.

Final preservation outcome is recorded in `10_STATE.md` and `evidence/preservation_final.json`.

## Exact primary PDF identity

- Title: *Guaranteed Tensor Luminality from Symmetry: A PT-Even Palatini Torsion Framework*.
- Author: Chien-Chih Chen. Journal: Symmetry 2026, 18, 170. DOI: `10.3390/sym18010170`.
- Local filename: `symmetry-18-00170-v2.pdf`; **50 pages**, **1,107,105 bytes**.
- SHA-256: `{hashes['symmetry-18-00170-v2.pdf']}`.
- Publisher v2 PDF downloaded read-only from [MDPI publication artifact](https://mdpi-res.com/d_attachment/symmetry/symmetry-18-00170/article_deploy/symmetry-18-00170-v2.pdf) into `evidence/publisher_v2.pdf`: **byte-identical** to local primary source.
- DOI metadata: `evidence/crossref_metadata.json`, retrieved from [Crossref DOI record](https://api.crossref.org/works/10.3390/sym18010170).
- Paper page 1: received 20 December 2025; revised 9 January 2026; accepted 13 January 2026; published 16 January 2026. Embedded creation/modification dates are 21 January 2026. Those PDF processing dates do not imply a different scientific edition. No claim is made about an undocumented earlier v1 or later revision.
- PDF text: `evidence/paper_page_01.txt` through `paper_page_50.txt`, plus `paper_full.txt`. Selected pages 31,32,34,35 also have independently rendered page evidence. Equation transcriptions on pp.31–34 were checked visually, not inferred from filenames.

## Environment

Existing Python: `{state['executable']}`, version `{sys.version.split()[0]}`. Platform: `{state['platform']}`.

**MISSING_ENVIRONMENT:** the project declares Python >=3.10, and environment.yml requests 3.11. The existing system interpreter with the required scientific libraries is 3.9.6. It can execute the existing 44 collected tests, but cannot execute one figure module's unpostponed `dict | None` annotations. Baseline runs below are explicitly compatibility-limited diagnostics, not certification of a supported environment. An existing Blender-bundled Python 3.11.11 has NumPy 1.26.4 but lacks SymPy, matplotlib, YAML, pytest and pandas. Two discovered virtual environments have broken interpreter links. No dependencies were installed or borrowed from another repository as authoritative scientific source.

Installed relevant package versions in the used environment:

|Package|Version|
|---|---|
'''+ '\n'.join(f'|{k}|{state["packages"].get(k,"MISSING_ENVIRONMENT") }|' for k in ['numpy','sympy','matplotlib','PyYAML','pytest','pandas','pypdf'])+'''

The complete installed distribution inventory is in `evidence/initial_state.json`. Requirements are lower bounds rather than an exact dependency lock; the historical smoke manifest records Python 3.12.3 and a different commit (`55f14b8`). This is a provenance mismatch, not proof that the currently audited source is corrupted.

## Evidence hierarchy

L0 published statement; L1 explicit derivation; L2 actual action-to-coefficient source derivation; L3 independent numerical reproduction; L4 configured/demo result; L5 figure regeneration; L6 documentation assertion. An L1 counterexample can defeat an L0 theorem even when its L4/L5 demonstrations reproduce. Integrity hashes certify bytes, not mathematical truth. Independent analytic tests below are L1 checks implemented with exact arithmetic; software-provenance probes remain software checks.
''')
write('01_SOURCE_AND_CLAIM_MAP.md',r'''
# Source and claim map

Locations are printed PDF pages (which equal file page numbers). Source lines refer to the preserved HEAD; the line-numbered complete snapshots `evidence/source_implementation.txt`, `source_tests.txt`, `source_scripts.txt`, `source_docs.txt`, `source_configs.txt`, `source_notebooks.txt` and the definition index give the traceable inventory. A dash means no implementing derivation/test/output was found, not that documentation supplies a proof.

## Assumptions and formal mathematics

|Claim|PDF location|Actual source/function|Configuration / original test|Figure/data|Evidence and obstruction|
|---|---|---|---|---|---|
|A1 domain/measure PT invariance|p.5 §2.6; pp.28–30 App.A|No domain/measure implementation; `spurion/pt_even.py:146 compute_parity` only named-string parity|Registry mapping; no test of integration domains|—|L0 assumption; L6 documentation. Can be consistently stipulated; arbitrary backgrounds not verified|
|A2 orientation and Hodge commutation|p.5; App.A A3,A9–A11|`pt_even.py` contains no Hodge tensor or orientation action|No Hodge test|—|L0 and conditional L1; App.A Hodge variation formula has independent counterexample|
|A3 projection/variation|p.6 Lemma1; p.29 A6|`pt_project` filters strings; `project_scalar:225` takes real part; no variational calculus|`test_algebra_basis.py` checks E→−D, not variation|—|L1 conditional equivariance; L2 absent; odd fixed background requires domain preservation|
|A4 admissible improvements|p.5; p.30 A12|`algebra/ibp.py:49 ibp_reduce` replaces one scalar symbol; no currents/boundary calculus|`test_algebra_basis.py:4`; `test_flux_ratio.py` tests a chosen curve|`fig_flux_ratio.py` → flux_ratio.csv|L4/L5; no derivation of actual route currents|
|A5 ordinary Nieh–Yan boundary|p.5; §5; App.C|No exterior-geometry NY implementation; `cspp_chain.order2_raw` reads JSON|cspp.json; C2 test uses different public fixed provider|C2 residual data|Ordinary exact-form identity L1; nonzero CS+ bulk not derived|
|A6 two derivatives/fixed spurion|p.5; Eq13 pp.8–9|`field_eq.solve_torsion_from_spurion` takes gradient as input|spurion alpha; C1 tests|C1 figures|L0 explicit truncation, L4 realization. Fixed background alone does not establish GR constraint algebra|
|Theorem1 scalar selection rules|p.6 Eq6; App.A|`compute_parity`, `pt_project`, `project_scalar`|No full geometric PT test; numeric projector is only Re|—|L1 elementary involution identity; L2 implementation limited|
|Proposition A1 product/self-adjointness claim|p.28 EqA2|No implementation|No original test|—|L0 false product identity; independent odd-function counterexample|
|Lorentz irrep projectors|pp.31–32 A13–A21|`torsion_decomp.py:55 trace_vector,:68 axial_vector,:90 pure_trace_piece,:101 axial_piece,:117 decompose`|test_torsion_decomp only pure trace|c1_components.csv|Published trace/axial projector signs and axial norm inconsistent; source axial index raising incomplete|
|Single-vector no-go|pp.11,32–33 Prop.A3 A23–A28|No representation-theory implementation; basis.py is monomial bookkeeping|No original no-go test|—|L1 survives for Lorentz-covariant linear maps built only from g,epsilon and one vector|

## Main physical chain

|Claim|PDF location|Actual source/function|Input / original test|Figure script → data / artifact|Exact evidence and missing source|
|---|---|---|---|---|---|
|C1 pure trace|pp.10–12 Thm2 Eqs16–24; App.B|`field_eq.py:38 solve_torsion_from_spurion` assigns T_vec=alpha*d_eps; `connection.py:35 torsion_from_trace`; `c1_pure_trace.run_c1`|alpha default1 or supplied; `test_c1_torsion:22,:38`; `test_torsion_decomp:12`|fig_c1_pure_trace → c1_components.csv, fig1_c1_pure_trace.pdf|L4 construct/decompose check, not action variation. Full EH torsion contribution missing in paper Step2|
|C1 alignment|p.13 Fig4|`c1_pure_trace.py:42 alignment_samples` zeros; `run_c1` angle=0|200 samples; same C1 tests|fig_c1_alignment → c1_alignment.csv, fig2_c1_alignment.pdf|L4 assigned angles; anti-alignment alpha<0 and null/Lorentzian angular domains not computed|
|C1 IT contraction|p.12 Eq24; p.33 A29|No general contraction derivation in field_eq|C1 tests check absent irreps, not Eq17 from action|same|Independent L1 contraction valid if pure trace imposed; no L2 determination of eta|
|C2 ROD|p.14 Eq26; p.34 A31–A33|`dbi_chain.py:7 order2_raw` reads dbi.json; public `equivalence/__init__.py:31 dbi_order2` returns .125|dbi.json [.125,0]; `test_c2_equivalence` actually exercises public constants|fig_c2_coeff_compare → c2_residuals.csv, fig3_c2_coeff_compare.pdf|ROD determinant algebra L1 verified independently; implementation L4|
|C2 CM|p.14; p.34 A34|`closed_metric_chain.order2_raw`; public `closed_metric_order2` constant plus nominal total_derivative key|closed.json [.125,0]; C2 tests drop unlisted key|same|L4. Actual metric determinant differs from claimed traceless determinant|
|C2 CS+|p.14; p.34 A35|`cspp_chain.order2_raw`; public `cspp_order2` adds ~1e−16 offsets|cspp.json [.125,0]; C2 tests|same|L4; no explicit nonzero projected operator or derivation|
|C2 equivalence / improvements|p.14 Prop1 Eqs27–29; pp.35–36 A36–A41|`coeff_extractor:25 coeff_vectors,:32 residual_norm,:45 reduce_to_IT`; `order2_checker:8 residual_scan`; `flux_ratio:20 flux_ratio_FRW`|ibp_tol unused in coefficients; notebooks/10 calls same return formula for all routes; test_flux_ratio uses 1+c R^(−sigma)|fig_c2_coeff_compare; fig_flux_ratio → flux_ratio.csv, fig9_flux_ratio.pdf|L4/L5; route densities not extracted, currents not integrated, flux curve imposed|
|C3 mixing coefficients|pp.16–17 Eqs30–34; pp.36–39 A42–A51 TablesA1,A2|`scripts/mixing_matrix_extract.py:64 load_mu_from_json,:72 locking_ratios,:83 write_outputs`|mixing_matrix_FRW.json; test_mixing_matrix_extract:11,23,43|make mixing → mixing_matrix.csv, mixing_matrix_meta.json +MD5|L4 staging and arithmetic. Tables contain symbolic labels, not evaluated coefficients. SOURCE_NOT_DERIVED for all four entries|
|C3 locking existence/uniqueness|p.17 Eq33 and text; p.37 Prop.A4|`gw/locking.py:42 delta_ct2,:61 apply_locking` changes one coefficient to force chosen dot product zero; `apply:17` simply sets locked=True|test_c3_tensor:86,150; no original rank/nullspace test|locking_curve y=−x over heatmap|L4; chosen weights +.25,−.5,+.125 not from ADM; det≠0 theorem contradicts nonzero kernel|
|C3 K=G / speed|pp.17,19 Eqs35–40; p.38 A52–A55|`quadratic_action.py:23 cT2_of_k`, `tensor_action_coeffs:87`; `tensor_mode.py:45 _ct_from_coeffs_or_config`|locked=True gives ones or (1,1); test_c3_tensor, test_paper_mode_apis|fig_c3_dispersion → dispersion.csv; fig_c3_cT_heatmap → cT_grid.npz|L4 hardcoded phenomenology, no physical K(w),G(w) Hessians|
|C3 waveform|p.19 Fig9|`tensor_mode.py:88 waveform_overlay` sets both sin(k*t)|wave number .05; no waveform evolution test|fig_gw_waveform_overlay → waveform_overlay.csv|L4/L5 identical analytic arrays, not integrated GW equations|
|C3 DoF / Hamiltonian|p.20 §7.2 Fig10|`degeneracy.py:13 principal_eigs,:29 principal_symbol_eigs,:43 hamiltonian_eigs`; no Poisson brackets|test_c3_tensor:118; test_paper_mode_apis checks two set entries|fig_c3_degeneracy → deg_eigvals.csv, fig6_c3_degeneracy.pdf|L4 synthetic eigenvalues; physical constraints missing; figure uses alias synthetic n=2 unlocked array|
|NLO complete basis / unique direction|p.9 Table2; §8 pp.21–23; App.H pp.46–49 A72–A93|`gw/nlo.py:29 basis_labels_nlo,:34 coeffs_from_config,:65 delta_KG,:93 predict_offsets`|test_nlo:10,19,28 checks labels/zero/k²; names are gradT_sq_eff,Ricci_deps_deps_eff, not Weyl² table|fig_nlo_offsets → nlo_offsets.csv, fig8_nlo_offsets.pdf|L4 parameterization; no four-derivative TT reduction; dimensional/operator-basis mismatch|

## Other inspected implementation and notebooks

`algebra/indices.py` defines commuting placeholders D,E,TdotDeps,T2; `algebra/basis.py` projects their coefficient vectors; `bianchi.py:28 apply_bianchi` returns its argument unchanged. They do not construct an invariant tensor basis or establish completeness. `numerics/grids.py`, `solvers.py`, `validate.py` contain generic grid/linear-algebra/tolerance helpers; no physical action-to-Hessian derivation is hidden there. The source `ArrayLike = np.ndarray | Iterable[float]` in solvers is also a Python>=3.10 runtime requirement, even though not on the baseline tested path.

`spurion/profile.py` implements constant/plane/Gaussian/custom externally specified profiles and numerical derivatives, not a spurion equation of motion. `io/config.py` loads configuration, creates output directories, and writes a manifest only when called by the CLI. `cli.py` remains a smoke/stub interface: `figs --which all` still makes a smoke artifact, while c1/c2/c3 commands print stub messages. It is not equivalent to `make paper`.

All four `.ipynb` files were inspected as JSON, contain zero saved outputs, and mirror the `.py` demos. 00 calls default tensor_action_coeffs returning(1,1); 10 calls a route-independent formula; 20 plots imposed flux convergence; 30 plots imposed k² offsets. Running notebook installation cells was deliberately excluded; no notebook is an independent derivation.

## Documentation comparison (L6)

|Document|Assertion|Observed source behavior|
|---|---|---|
|README.md|Full end-to-end C1/C2/C3, 44 passed, pinned environment, all figures|44 collected but 42 run +2 skip here; version lower bounds not a lock; physics stages are model/demo APIs; full figure run environment-limited|
|docs/REPRODUCIBILITY.md|make paper-test/paper/mixing verifies core claims; paper mode disables fallback|Outer script fallback is guarded; installed core toy APIs pass unchanged; make mixing only stages inputs; historical manifest not refreshed by make paper|
|docs/REVIEWER_NOTES.md|Not just plotting; uniqueness/equivalence/DoF tested; single strongest audit point make mixing|C1 constructs pure trace, C2 compares constants, C3 returns chosen speed/eigenvalues; no physical mixing extraction|
|docs/FAQ.md|Full pipeline and tensor structural identities reproduced|Reproducible programmed assumptions, no action-to-observable chain|
|docs/ADOPTION_PACK.md|Symmetry rather than tuning; unique NLO slope2 forecast|No demonstrated PT-specific enforcement; slope2 follows assumed k² input|

Paper figure numbering differs from repository numbering (e.g. PDF Fig2 corresponds to repo fig1; PDF Fig10 to repo fig6). Matching uses script names/captions, not identical figure numbers. PDF title/version identity is exact; this naming mismatch is not treated as a scientific counterexample.

## Dependencies and completeness boundary

All files identified in the task are present. Missing elements are mathematical/physical provenance (evaluated mixing entries, ADM densities, CS+ definition, full constraint algebra), not missing imports that installation could fix. Compatible full runtime remains MISSING_ENVIRONMENT as documented in 00. No other repository was used as an authoritative replacement.
''')
write('02_BASELINE_REPRODUCTION.md',r'''
# Baseline reproduction

## Results and environment qualification

The original historical assertion is **44 passed**. Current unchanged sources collect 44 cases; the diagnostic execution in the existing Python3.9.6 environment yields **42 passed, 2 skipped**. Both skips are CLI tests guarded by `shutil.which("palpt") is None`. No original source/test/configuration was patched and no package was installed to force that count.

|Execution in working_copy|Exit|Result|Evidence|
|---|---:|---|---|
|`python3 -m pytest -q -rs --basetemp=...`|0|42 passed,2 skipped,394 warnings;10.46s|baseline.stdout.txt/.stderr.txt|
|`make paper-test`|2|Shell cannot locate pytest (underlying command127)|paper_test.stdout.txt/.stderr.txt|
|`PALPT_REQUIRE_REAL_APIS=1 python3 -m pytest -q -rs --basetemp=...`|0|42 passed,2 skipped,394 warnings;1.67s|paper_test_module.stdout.txt/.stderr.txt|
|`make mixing PYTHON=/Library/Developer/CommandLineTools/usr/bin/python3`|0|Ratios both0.666666666667; JSON staged|mixing.stdout.txt/.stderr.txt|
|`make paper PYTHON=/Library/Developer/CommandLineTools/usr/bin/python3`|2|First eight modules execute; final flux module import fails on unpostponed dict\|None annotation|paper.stdout.txt/.stderr.txt|

Commands, working directories, environment overrides, elapsed times and exit codes are in `evidence/baseline_commands.json`, `extra_commands.json`; runners are retained. Project requires Python>=3.10. **MISSING_ENVIRONMENT** prevents a supported-version full reproduction. The `-m pytest` paper-mode rerun bypasses only executable discovery, without changing scientific code. Neither the unsupported-version figure failure nor the missing CLI entrypoint is counted as a scientific theorem failure. A complete 44-pass historical baseline is **not** claimed.

No notebook execution or installation cells were used. Optional jupytext/jupyter/Snakemake dev tooling is absent from the used environment; those optional pipelines were not required or installed.

## What the tests actually exercise

- C1 (3 cases): generate already pure-trace tensors and decompose them; random tests vary external gradients and a positive assigned alpha. No independent action variation or eta derivation.
- C2 (10 cases including six alpha parameterizations): public fixed-provider API, placeholder basis ordering, dropping an unlisted `total_derivative` key, residual monotonicity for deliberately inserted 1e−16 noise. `alpha` is ignored by the providers. These tests do not use the three JSON-reading `*_chain.py` routes.
- C3 (7): assigned coefficient dot product and Z model; `lock=True` directly returns ones; a coefficient is edited by apply_locking to annihilate the chosen dot product; synthetic two-number arrays are checked for sign. No physical kinetic/gradient Hessian or constraint algebra.
- Mixing (3): read manually supplied matrix; calculate two ratios; test success/failure on their agreement. Useful API validation, not second variation.
- NLO (3): ASCII labels, zero parameter limit, and expected k² scaling of an expression defined as k².
- Flux (2): positivity of radius and approach to1 of `1+c R^(-sigma)`.
- Algebra (1): actual but narrow scalar IBP identity E→−D.
- Paper API (2): presence of functions and assigned luminal arrays/eigenvalues.
- Scripts (8): files/MD5 sidecars exist for seven figure modules plus smoke orchestration. They do not validate plotted science. NLO/flux figure modules are not individually covered here.
- CLI (2 skipped): entrypoint version/smoke output, no physics.

## Runtime call graphs

Full observed edges and results are in `evidence/runtime_call_graphs.json` (generated by Python call profiling, restricted to copy source).

```
C1 test/driver -> run_c1 -> solve_torsion_from_spurion
              -> T_vec = alpha*d_eps -> torsion_from_trace
              -> check_pure_trace -> decompose
              -> angle_rad = 0

C2 original tests -> equivalence.__init__.compute_equivalence_report
                 -> dbi_order2/closed_metric_order2/cspp_order2 (constants)
                 -> coefficient differences (~0 or 1e-16)
C2 figure -> order2_checker.residual_scan -> coeff_extractor.residual_norm
          -> coeff_vectors -> three order2_raw JSON readers -> pairwise norms
C2 notebook -> reduce_to_IT(route=...) -> same lam²/8*(-6 eta² Seps) for every route

C3 make mixing -> load_mu_from_json -> locking_ratios -> write_outputs
               -> ratio_N==ratio_divN -> selected w_ROD/w_CM
C3 speed -> cT_of_k -> _ct_from_coeffs_or_config -> ones_like(k) if lock=True
C3 waveform -> waveform_overlay -> sin(k*t), sin(k*t)
C3 figure eigenvalues -> hamiltonian_eigs -> principal_symbol_eigs
                     -> principal_eigs(locked=False,n=2) -> synthetic [1.0016,1.002]
```

There is no observed arrow from Eq13 variation to C2 JSON, from ADM to mixing JSON, from mixing ratio to physical K,G, or from physical K,G to the plotted waveform.

## Strict flag and fallbacks

`PALPT_REQUIRE_REAL_APIS=1` is read once at import in the seven principal figure scripts. It prevents their final outer numerical fallback when an API search fails (independently tested by an injected missing-API exception). It does **not** reject demo implementations residing inside the nominal APIs:

- locked cT and K=G assignments are accepted;
- JSON coefficient comparisons are accepted;
- `hamiltonian_eigs(config=...)` leads to an internal synthetic two-entry fallback;
- real-API tests only assert names and prescribed outputs;
- make_all_figs config-loader fallback and CLI smoke fallback are not globally guarded by this flag.

No outer numerical fallback is needed for the eight modules completed by make paper. The degeneracy adapter first tries incompatible `principal_symbol_eigs(config=...)`, catches the TypeError, then succeeds through `hamiltonian_eigs`; this is an API alias change, not an action-derived spectrum. Its synthetic output matches the two entries already stored in the original data.

## Artifact consistency

`evidence/artifact_consistency.json` records all sidecar checks and before/after copy hashes.

- All **20 original MD5 sidecars** match their adjacent artifacts; all checked post-run copy sidecars also match. Naked digest files are not standard `md5sum -c` filename records, despite the documentation's sample command.
- 24 figure-tree paths changed during mixing/partial-paper generation (including sidecars and timestamp-dependent PDFs/meta).
- C2 residuals, C1 angles, NLO offsets, and the two synthetic degeneracy entries numerically reproduce exactly.
- Dispersion differs by at most6.94e−18; waveform by1.11e−16 under the current numerical environment. Model and GR waveform arrays remain identically constructed.
- The flux CSV is unchanged because its final module never ran. Its equal hash is **not** counted as regeneration.
- Mixing CSV stages the same numeric inputs; metadata timestamp and paths are runtime data, so its hash is not expected to remain identical.
- PDF byte matching is not a reliable numerical criterion across environments because render metadata and libraries change. Matching sidecars only prove self-consistent outputs.
- `figs/data/run_manifest.json` still describes a smoke run at commit55f14b8, Python3.12.3, dated2025-09-21. `make paper` does not write a new manifest. It cannot establish provenance for the audited paper run.

All original artifacts remained untouched. See 10_STATE for final byte verification.
''')
