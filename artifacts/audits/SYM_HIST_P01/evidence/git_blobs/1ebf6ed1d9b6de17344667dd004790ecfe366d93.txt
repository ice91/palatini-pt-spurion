# Adoption Pack — palatini-pt-spurion (PT-even Palatini torsion, cT=1 by construction)

**What this is:** a reproducible codebase that regenerates all paper figures/data and verifies key structural identities (C1/C2/C3) end-to-end.

- Paper (open access): Symmetry, “Guaranteed Tensor Luminality from Symmetry: A PT-Even Palatini Torsion Framework”
  DOI: **10.3390/sym18010170**
- Repo: https://github.com/ice91/palatini-pt-spurion
- YouTube (paper walkthrough): https://youtu.be/2U4QbwHlbq4

---

## 60-second overview (for readers)
**Problem:** Multimessenger observations constrain the GW speed to be luminal. Many modified gravity models survive only via parameter tuning.  
**Claim:** In this metric-affine (Palatini) torsion framework, **tensor luminality at quadratic order** follows from **symmetry-selected structure**, not tuning.

**Key structural ingredients**
1) **PT-even scalar projection**: keeps relevant scalar densities real and parity-even.  
2) **Projective invariance** via a **non-dynamical Stueckelberg-type compensator**, entering only through its gradient.

**Core results implemented and tested**
- **(C1)** torsion uniquely reduces to **pure trace**, aligned with compensator gradient (algebraic uniqueness)
- **(C2)** equivalence (in the bulk, up to improvement terms) among three quadratic constructions:
  **rank-one determinant route**, **closed-metric deformation**, **PT-even CS/Nieh–Yan route**
- **(C3)** coefficient-locking identity implies **K=G** for tensor modes on admissible domains → **two polarizations, cT=1**

**Falsifiable next-to-leading prediction**
- \(\delta c_T^2(k)= b\,k^2/\Lambda^2\) for \(k\ll\Lambda\)
- implies **slope-2** scaling in log–log fits across frequency bands (**PTA / LISA / LVK**)

---

## 5-minute reproduction (recommended)
### Option A — conda (recommended)
```bash
conda env create -f environment.yml
conda activate palpt
make paper
make paper-test
````

### Option B — venv/pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
make paper
make paper-test
```

**Expected:** PDFs under `figs/pdf/` and datasets under `figs/data/`, plus **44 passed**.

---

## Artifacts you can cite / inspect

- Figures: `figs/pdf/fig*.pdf`
    
- Data: `figs/data/*.csv`, `figs/data/*.npz`
    
- Integrity: `.md5` sidecars next to each artifact
    
- Run manifest: `figs/data/run_manifest.json`
    
- App. D artifact (locking check):
    
    - `figs/data/mixing_matrix.csv`
        
    - `figs/data/mixing_matrix_meta.json`
        

---

## “Trust but verify” checklist (for reviewers)

1. Run the exact paper test suite: `make paper-test`
    
2. Rebuild paper artifacts: `make paper`
    
3. Verify App. D lock: `make mixing` (fails if not locked)
    
4. Hash integrity: compare each artifact with its `.md5` sidecar
    

---

## Quick citation

- See `CITATION.cff` (software citation).
    
- Paper DOI: **10.3390/sym18010170**
    
- If you use the code or artifacts, please cite **both** the paper and the software snapshot.
    

---

## Contact

- Author: Chien-Chih Chen ([rocky@cht.com.tw](mailto:rocky@cht.com.tw))
    
- Issues: please use GitHub Issues (templates included)
    