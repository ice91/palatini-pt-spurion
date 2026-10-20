# Reproducibility Guide — palatini-pt-spurion

This document is a step-by-step guide to reproduce the paper artifacts and verify the core claims (C1/C2/C3) exactly as used in the published Symmetry article (DOI: 10.3390/sym18010170).

Repo: https://github.com/ice91/palatini-pt-spurion  
YouTube walkthrough: https://youtu.be/2U4QbwHlbq4

---

## 0) What you should get when everything works

After a full run, you should see:

- Paper figures (PDF): `figs/pdf/fig1_*.pdf` ... `figs/pdf/fig9_*.pdf`
- Data artifacts: `figs/data/*.csv`, `figs/data/*.npz`, plus `.md5` sidecars
- Test suite: **44 passed**

Sanity indicator:
- `figs/png/fig_smoke.png` exists (quick smoke output)
- `figs/data/run_manifest.json` exists (records run inputs/versions)

---

## 1) Environment setup

### Option A — conda (recommended)
```bash
conda env create -f environment.yml
conda activate palpt
````

### Option B — venv/pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

Optional but recommended:

```bash
pre-commit install
pre-commit run -a
```

---

## 2) Reproduce all paper figures & data

Run:

```bash
make paper
```

Notes:

- This runs in “paper mode”: `PALPT_REQUIRE_REAL_APIS=1` and `PYTHONHASHSEED=0`
    
- “paper mode” disables fallbacks to ensure the run is strict and deterministic.
    

Outputs:

- PDFs: `figs/pdf/fig*.pdf`
    
- Data: `figs/data/*`
    
- Hash sidecars: `*.md5`
    

---

## 3) Verify the full paper test suite

Run:

```bash
make paper-test
```

Expected (approx):

- `44 passed` in a few seconds (machine-dependent).
    

---

## 4) App. D artifact: mixing matrix + coefficient locking

Run:

```bash
make mixing
```

This regenerates:

- `figs/data/mixing_matrix.csv`
    
- `figs/data/mixing_matrix_meta.json`
    

and enforces the lock condition (fails if not satisfied).

---

## 5) Verify artifact integrity with MD5 sidecars

Artifacts in `figs/` have `.md5` files next to them.

Example check:

```bash
cd figs/pdf
md5sum -c fig1_c1_pure_trace.pdf.md5
```

Or check all (Linux):

```bash
find . -name "*.md5" -print0 | xargs -0 -n 1 md5sum -c
```

(macOS uses `md5`, not `md5sum`; if needed you can install coreutils or verify manually.)

---

## 6) “Smoke” run for quick confidence

If you only want a fast check:

```bash
make figs
```

This generates:

- `figs/png/fig_smoke.png`
    
- and minimal supporting intermediates
    

---

## 7) Notebook execution (optional)

Sync (Jupytext):

```bash
make nb
```

Execute notebooks headless:

```bash
make nb-test
```

Colab is supported for viewing/running notebooks without `snakemake/graphviz`.

---

## 8) Snakemake pipeline DAG (optional)

To build the pipeline dependency graph:

```bash
make dag
```

Requirements:

- `snakemake` installed (via conda or pip)
    
- system `graphviz` (`dot`) installed
    

Output:

- `tex/snakemake_dag.pdf`
    

---

## 9) Troubleshooting

### A) `PALPT_REQUIRE_REAL_APIS=1` causes failures

This is intentional in paper mode. It means a fallback path was attempted.

- Inspect the traceback
    
- Ensure the environment matches (conda env or `pip install -e .[dev]`)
    
- Re-run `make paper-test` to isolate
    

### B) Missing `dot` or Snakemake for `make dag`

Install graphviz:

- Ubuntu: `sudo apt-get install graphviz`
    
- macOS: `brew install graphviz`
    
- conda: `conda install -c conda-forge graphviz`
    

Install snakemake:

- conda: `conda install -c conda-forge snakemake`
    
- pip: `pip install snakemake`
    

### C) Matplotlib + LaTeX strings

When you use LaTeX-like labels in titles/axes, prefer raw strings:

- `r"$\mathcal R_{X/Y}$"`
    

---

## 10) What “C1 / C2 / C3” map to in the code

This is a practical map (not a theory summary):

- C1 (pure-trace torsion uniqueness)
    
    - `palatini_pt/palatini/c1_pure_trace.py`
        
    - tests: `tests/test_c1_torsion.py`
        
- C2 (three-chain equivalence at quadratic order)
    
    - `palatini_pt/equivalence/*_chain.py`
        
    - tests: `tests/test_c2_equivalence.py`
        
    - notebook demo: `notebooks/10_c2_symbolic_demo.ipynb`
        
- C3 (tensor sector locking: K = G → cT = 1)
    
    - `palatini_pt/gw/locking.py`, `palatini_pt/gw/tensor_mode.py`
        
    - tests: `tests/test_c3_tensor.py`
        
    - figures: `scripts/fig_c3_*.py`
        

---

## 11) Minimal “reviewer script”

If you only run three commands, run these:

```bash
make paper-test
make paper
make mixing
```

That is sufficient to verify:

- unit/integration test coverage (paper suite)
    
- figure/data regeneration
    
- App. D locking check
    

---

## Citation

Please cite:

- the companion paper (DOI: 10.3390/sym18010170)
    
- and this codebase (see `CITATION.cff`)
    

License: MIT (see `LICENSE`)
