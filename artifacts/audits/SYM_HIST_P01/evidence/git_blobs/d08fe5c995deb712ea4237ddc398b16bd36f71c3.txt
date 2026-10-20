# Reviewer Notes — palatini-pt-spurion (DOI: 10.3390/sym18010170)

This document is a **reviewer-oriented index** that maps the paper’s key claims (C1/C2/C3 + NLO forecast) to:
- **where they live in the code**
- **which artifacts to inspect**
- **what commands reproduce/verify them**

Repo: https://github.com/ice91/palatini-pt-spurion  
Paper (open access): https://www.mdpi.com/2073-8994/18/1/170 (DOI: 10.3390/sym18010170)  
YouTube walkthrough (paper + code): https://youtu.be/2U4QbwHlbq4

---

## 0) TL;DR — minimal reviewer workflow (3 commands)

After installing the environment (see Section 1), these three commands are sufficient to verify the reproducibility and the core identities:

```bash
make paper-test
make paper
make mixing
````

Expected:

- `make paper-test` → **44 passed**
    
- `make paper` → regenerates all paper figures/data under `figs/pdf/` and `figs/data/`
    
- `make mixing` → regenerates App. D artifacts and **fails if the locking condition is violated**
    

---

## 1) Environment setup (recommended)

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate palpt
```

### Option B — venv/pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

Sanity:

```bash
palpt --help
pytest -q
```

---

## 2) What the repo is claiming (reviewer framing)

This repository is not only a figure generator. It is designed to make the paper’s structural claims **auditable**:

- **C1** (pure-trace torsion uniqueness): torsion is forced into a pure-trace form, aligned with the compensator gradient (algebraic uniqueness).
    
- **C2** (three-chain equivalence at quadratic order): three quadratic constructions are bulk-equivalent up to improvement/boundary terms.
    
- **C3** (tensor-sector locking): the coefficient identity locks **K = G** for the tensor sector, ensuring **two tensor polarizations** and **cT = 1** at quadratic order.
    
- **NLO / falsifiable forecast**: a small-(k) expansion yields a **slope-2** scaling (toy forecast tooling included as a demonstrator).
    

This repo implements the above as:

- paper-checkable identities (symbolic/algebraic checks),
    
- numerical grids (where relevant),
    
- artifact generation with integrity hashes (`.md5`),
    
- and a test suite covering algebra + equivalence + GW sector + script outputs.
    

---

## 3) Where to look: code map for C1/C2/C3

### C1 — pure-trace torsion uniqueness

**Core module(s):**

- `palatini_pt/palatini/c1_pure_trace.py`
    
- `palatini_pt/palatini/torsion_decomp.py`
    
- `palatini_pt/palatini/connection.py`
    

**Primary tests:**

- `tests/test_c1_torsion.py`
    
- `tests/test_torsion_decomp.py`
    

**Paper figure(s) most closely aligned:**

- `figs/pdf/fig1_c1_pure_trace.pdf`
    
- `figs/pdf/fig2_c1_alignment.pdf`
    

**Generation scripts:**

- `scripts/fig_c1_pure_trace.py`
    
- `scripts/fig_c1_alignment.py`
    

---

### C2 — three-chain equivalence (quadratic order, bulk equivalence)

**Core module(s):**

- `palatini_pt/equivalence/dbi_chain.py`
    
- `palatini_pt/equivalence/closed_metric_chain.py`
    
- `palatini_pt/equivalence/cspp_chain.py`
    
- `palatini_pt/equivalence/order2_checker.py`
    
- `palatini_pt/equivalence/coeff_extractor.py`
    

**Primary tests:**

- `tests/test_c2_equivalence.py`
    

**Notebook demo (optional, explanatory):**

- `notebooks/10_c2_symbolic_demo.ipynb` (and `.py`)
    

**Paper figure(s) most closely aligned:**

- `figs/pdf/fig3_c2_coeff_compare.pdf`
    

**Generation scripts:**

- `scripts/fig_c2_coeff_compare.py`
    

---

### C3 — tensor sector (locking K = G → cT = 1)

**Core module(s):**

- `palatini_pt/gw/locking.py`
    
- `palatini_pt/gw/tensor_mode.py`
    
- `palatini_pt/gw/quadratic_action.py`
    
- `palatini_pt/gw/degeneracy.py`
    
- `palatini_pt/numerics/grids.py`
    
- `palatini_pt/numerics/solvers.py`
    
- `palatini_pt/numerics/validate.py`
    

**Primary tests:**

- `tests/test_c3_tensor.py`
    
- `tests/test_nlo.py` (for NLO tooling)
    
- `tests/test_paper_mode_apis.py` (paper-mode strictness)
    
- `tests/test_scripts_figs.py` (script outputs)
    

**Paper figure(s) most closely aligned:**

- `figs/pdf/fig4_c3_cT_heatmap.pdf`
    
- `figs/pdf/fig5_c3_dispersion.pdf`
    
- `figs/pdf/fig6_c3_degeneracy.pdf`
    
- `figs/pdf/fig7_gw_waveform_overlay.pdf`
    
- `figs/pdf/fig8_nlo_offsets.pdf`
    
- `figs/pdf/fig9_flux_ratio.pdf`
    

**Generation scripts:**

- `scripts/fig_c3_cT_heatmap.py`
    
- `scripts/fig_c3_dispersion.py`
    
- `scripts/fig_c3_degeneracy.py`
    
- `scripts/fig_gw_waveform_overlay.py`
    
- `scripts/fig_nlo_offsets.py`
    
- `scripts/fig_flux_ratio.py`
    

---

## 4) Paper artifacts: what gets generated and where

### Figures

- Output directory: `figs/pdf/`
    
- Paper figures: `figs/pdf/fig1_*.pdf` … `figs/pdf/fig9_*.pdf`
    
- A quick smoke output is also available under `figs/png/fig_smoke.png`
    

### Data artifacts

- Output directory: `figs/data/`
    
- CSV / NPZ artifacts used to build figures:
    
    - `c1_alignment.csv`, `c1_components.csv`
        
    - `c2_residuals.csv`
        
    - `cT_grid.npz`, `dispersion.csv`, `deg_eigvals.csv`
        
    - `waveform_overlay.csv`, `nlo_offsets.csv`, `flux_ratio.csv`
        
    - plus metadata and run manifest: `run_manifest.json`
        

### Integrity hashes

- Every key artifact has an adjacent `.md5` sidecar in `figs/pdf/` and `figs/data/`
    
- These are meant to make “trust but verify” easy.
    

---

## 5) Reproducing everything (exact paper path)

### A) Regenerate all figures + data (paper grids)

```bash
make paper
```

This invokes paper-mode settings:

- `PALPT_REQUIRE_REAL_APIS=1`
    
- `PYTHONHASHSEED=0`
    

Meaning:

- strict mode to prevent silent fallbacks,
    
- reduced nondeterminism where possible.
    

### B) Run the exact paper test suite

```bash
make paper-test
```

Expected:

- `44 passed`
    

---

## 6) App. D: mixing matrix artifact + locking check (high-value audit point)

This is the most “audit-friendly” entry point for the tensor-sector locking mechanism.

Run:

```bash
make mixing
```

Outputs:

- `figs/data/mixing_matrix.csv`
    
- `figs/data/mixing_matrix_meta.json` (includes tolerance and lock diagnostics)
    
- corresponding `.md5` files
    

Behavior:

- The command **fails** if the locking condition is not satisfied within tolerance.
    

Where it comes from:

- source JSON: `configs/coeffs/mixing_matrix_FRW.json`
    
- script: `scripts/mixing_matrix_extract.py`
    
- test: `tests/test_mixing_matrix_extract.py`
    

---

## 7) Verifying artifact integrity (MD5 sidecars)

### Linux (md5sum)

To check a single file:

```bash
cd figs/pdf
md5sum -c fig4_c3_cT_heatmap.pdf.md5
```

To check all artifacts:

```bash
find .. -name "*.md5" -print0 | xargs -0 -n 1 md5sum -c
```

### macOS note

macOS ships `md5` (not `md5sum`). If you want `md5sum`, install coreutils (e.g., via Homebrew) or verify manually.

---

## 8) Optional: notebook-based demonstrations (not required for paper reproduction)

Notebooks are explanatory and optional:

- `notebooks/00_sanity.ipynb`
    
- `notebooks/10_c2_symbolic_demo.ipynb`
    
- `notebooks/20_flux_ratio.ipynb`
    
- `notebooks/30_nlo_offsets.ipynb`
    

Headless execution:

```bash
make nb-test
```

Sync via jupytext:

```bash
make nb
```

Colab links are provided in `README.md`.

---

## 9) Common reviewer questions (quick answers)

### Q: “Is this just plotting, or does it actually test the claims?”

A: The repo includes a **full test suite (44 tests)** that covers:

- algebraic building blocks,
    
- the C1 torsion decomposition/uniqueness,
    
- the C2 quadratic equivalence checks,
    
- the C3 tensor-sector locking,
    
- and figure-script execution consistency.
    

### Q: “What is the single strongest audit point?”

A: `make mixing` (App. D) because it produces a **concrete artifact** (`mixing_matrix.csv` + meta) and **fails** if locking is not met.

### Q: “What if paper mode fails?”

A: Paper mode is intentionally strict. Failures usually indicate:

- environment mismatch (dependency/version),
    
- or a code path attempting a fallback that paper mode forbids.  
    Run `make paper-test` first to localize the issue.
    

---

## 10) Citation

- Paper DOI: **10.3390/sym18010170**
    
- Software citation: see `CITATION.cff`
    

If you reuse the code or artifacts, please cite **both** the paper and the software snapshot.

License: MIT (see `LICENSE`)
