# FAQ — palatini-pt-spurion

This FAQ focuses on practical questions reviewers/readers typically ask when running the repo.

Repo: https://github.com/ice91/palatini-pt-spurion  
Paper (Symmetry): DOI 10.3390/sym18010170  
YouTube walkthrough: https://youtu.be/2U4QbwHlbq4

---

## Q1. What exactly does this repository reproduce?

It reproduces the full **C1/C2/C3 pipeline** and the **paper figures & data** end-to-end:

- **C1**: torsion is forced into a **pure-trace** form (algebraic uniqueness)
- **C2**: three quadratic routes are **bulk-equivalent** up to improvement terms
- **C3**: tensor-sector coefficient identity locks **K = G**, giving **cT = 1** with two propagating polarizations

Practical outputs:

- `figs/pdf/fig1_*.pdf` ... `figs/pdf/fig9_*.pdf`
- `figs/data/*.csv`, `figs/data/*.npz`
- `.md5` hash sidecars for artifacts
- `44` tests covering algebra + equivalence + GW sector + scripts

---

## Q2. What is the fastest way to check it works?

Run a smoke pipeline:

```bash
make figs
````

You should get:

- `figs/png/fig_smoke.png`
    

Then run tests:

```bash
make paper-test
```

Expected:

- `44 passed`
    

---

## Q3. What is the “one command” full reproduction?

```bash
make paper
```

This generates _all_ paper figures and the corresponding data artifacts.

---

## Q4. What is “paper mode” and why does it exist?

The `make paper` and `make paper-test` targets set:

- `PALPT_REQUIRE_REAL_APIS=1`
    
- `PYTHONHASHSEED=0`
    

This is a strict mode designed to prevent silent fallbacks and reduce accidental nondeterminism.

If paper mode fails, it usually means:

- an environment mismatch (missing dependency / wrong version), or
    
- a fallback path was attempted (which paper mode disallows).
    

---

## Q5. Where is the App. D mixing matrix and locking check?

Command:

```bash
make mixing
```

Outputs:

- `figs/data/mixing_matrix.csv`
    
- `figs/data/mixing_matrix_meta.json`
    

Behavior:

- It **fails** if the locking condition is not satisfied within tolerance.
    

---

## Q6. How do I know the artifacts are the “same run” as the paper?

Two layers:

1. **Re-generation** with `make paper` under a pinned environment.
    
2. **Integrity hashes**: `.md5` files alongside artifacts.
    

Example check:

```bash
cd figs/pdf
md5sum -c fig4_c3_cT_heatmap.pdf.md5
```

(If you're on macOS without `md5sum`, install coreutils or compare manually.)

---

## Q7. Are the figures produced by notebooks or scripts?

The “paper path” uses scripts and configs:

- scripts: `scripts/fig_*.py`, orchestrated by `scripts/make_all_figs.py`
    
- configs: `configs/paper_grids.yaml`, plus JSON coefficients under `configs/coeffs/`
    

Notebooks exist as explanatory demos and are optional:

- `notebooks/00_sanity.ipynb`
    
- `notebooks/10_c2_symbolic_demo.ipynb`
    
- `notebooks/20_flux_ratio.ipynb`
    
- `notebooks/30_nlo_offsets.ipynb`
    

---

## Q8. Why do you include `.md5` files for PDFs and CSVs?

Because it makes the reproducibility claim **auditable**:

- Readers can confirm files have not been modified
    
- Reviewers can verify their run matches expected artifacts (or quantify differences)
    

---

## Q9. I got an error: “dot not found” or “snakemake not installed”.

This only affects the optional pipeline DAG:

```bash
make dag
```

Install:

- graphviz (`dot`)
    
- snakemake
    

Then:

```bash
make dag
```

Output:

- `tex/snakemake_dag.pdf`
    

---

## Q10. I got a plotting error with LaTeX-like labels.

Prefer raw strings in matplotlib labels:

- `r"$\mathcal R_{X/Y}$"`
    

Also, avoid requiring a full LaTeX installation unless you explicitly enable it.

---

## Q11. What should I cite if I reuse the code?

- Cite the **paper** (DOI: 10.3390/sym18010170)
    
- Cite the **software snapshot** per `CITATION.cff`
    

---

## Q12. What license is this released under?

MIT License — see `LICENSE`.

---

## Q13. Who is this for?

- Reviewers wanting a fast, auditable verification path
    
- Readers who want to reproduce figures/data
    
- Researchers who want to reuse the algebra/equivalence/GW tooling
    

If you only want a minimal reviewer workflow, run:

```bash
make paper-test
make paper
make mixing
```
