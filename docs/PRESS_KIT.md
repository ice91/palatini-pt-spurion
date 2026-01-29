# Press Kit — palatini-pt-spurion (PT-even Palatini torsion; cT = 1 by symmetry)

This page is a lightweight “press kit” you can paste into a project website, a Medium post, or a GitHub repo. It packages the **why / what / how to verify** in a non-technical but accurate way.

- Paper (open access): Symmetry — “Guaranteed Tensor Luminality from Symmetry: A PT-Even Palatini Torsion Framework”  
  DOI: **10.3390/sym18010170**
- Repo: https://github.com/ice91/palatini-pt-spurion
- YouTube (paper walkthrough): https://youtu.be/2U4QbwHlbq4

---

## 1) One-paragraph summary

**palatini-pt-spurion** is a reproducible research codebase accompanying the Symmetry paper (DOI: 10.3390/sym18010170). It implements a **metric-affine (Palatini) gravity** framework with a **PT-even scalar projection** and a **non-dynamical spurion (compensator)**. The key message is that **gravitational-wave tensor modes remain exactly luminal (cT = 1) at quadratic order by symmetry-selected structure**, rather than by parameter tuning. The repository regenerates all figures/data and provides a full test suite so the core claims are **auditable end-to-end**.

---

## 2) What problem does this address?

Modern multi-messenger observations place stringent constraints on the **speed of gravitational waves**. Many modified gravity models can satisfy those constraints only after **fine-tuning parameters** or restricting to narrow corners of their parameter space.

This project targets a different question:

> Can luminal tensor propagation arise as a **structural identity** enforced by symmetry and consistency conditions, rather than tuning?

---

## 3) What is the core idea (non-technical)?

The framework combines two structural ingredients:

1) **PT-even scalar projection**  
   A symmetry selection that keeps the relevant scalar densities **real** and **parity-even**.

2) **Projective invariance via a spurion (compensator)**  
   A non-dynamical Stueckelberg-type field enters **only through its gradient** and acts as a compensator for the projective symmetry of the Palatini connection.

These ingredients are not added to “fit a number.” They act as **selection rules** that constrain what the theory can do.

---

## 4) What are the headline results?

The repository implements and verifies three structural claims (C1/C2/C3) used in the paper:

- **C1 — Pure-trace torsion (uniqueness):**  
  torsion is forced into a pure-trace form and aligns with the compensator gradient.

- **C2 — Three-chain equivalence (quadratic order):**  
  three different quadratic constructions are bulk-equivalent up to improvement/boundary terms:  
  (i) rank-one determinant route, (ii) closed-metric deformation, (iii) PT-even CS/Nieh–Yan route.

- **C3 — Tensor-sector locking (cT = 1):**  
  a coefficient-locking identity implies **K = G** for tensor modes on admissible domains, yielding **two propagating polarizations** and **cT = 1** at quadratic order.

**Falsifiable next-to-leading prediction (NLO):**
- a small-\(k\) expansion produces a **slope-2** scaling (toy forecast tooling included), intended as a concrete target across frequency bands (**PTA / LISA / LVK**).

---

## 5) Why should anyone trust it?

This project is built to support “trust but verify.”

**Reproducibility features:**
- **One-command figure regeneration**: `make paper`
- **Exact paper test suite**: `make paper-test` (expected: **44 passed**)
- **Auditable artifacts**: `figs/pdf/` and `figs/data/`
- **Integrity hashes**: `.md5` sidecars next to key artifacts
- **Run manifest**: `figs/data/run_manifest.json` stores run inputs/versions
- **App. D artifact**: `make mixing` produces a mixing-matrix CSV + meta and fails if locking is violated

---

## 6) The 60-second “how to verify” (for reviewers and readers)

### Install (choose one)
```bash
# Option A: conda
conda env create -f environment.yml
conda activate palpt

# Option B: venv/pip
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
````

### Verify (3 commands)

```bash
make paper-test
make paper
make mixing
```

Expected:

- `44 passed`
    
- PDFs under `figs/pdf/`
    
- data artifacts under `figs/data/`
    
- mixing-matrix artifacts under `figs/data/` (with `.md5`)
    

---

## 7) Media-ready talking points (copy/paste)

**Short bullet version**

- “A reproducible Palatini gravity codebase showing cT = 1 by symmetry, not tuning.”
    
- “Implements C1/C2/C3 structural identities and regenerates all paper figures.”
    
- “Includes full tests + integrity hashes + an App. D mixing-matrix locking artifact.”
    
- “Provides a falsifiable slope-2 NLO target across PTA/LISA/LVK bands.”
    

**One-sentence version**

- “This repo makes the paper’s ‘cT = 1 by construction’ claim auditable via one-command reproduction, tests, and hashed artifacts.”
    

---

## 8) Visual assets

A publication certificate and a banner are available to corresponding authors from the MDPI submission system (login required).  
This repo also contains ready-to-share figures under `figs/pdf/` once generated.

---

## 9) Citation

If you use this software or its artifacts, please cite:

- the paper: DOI **10.3390/sym18010170**
    
- the software snapshot: see `CITATION.cff`
    

---

## 10) Contact

- Author: Chien-Chih Chen — [rocky@cht.com.tw](mailto:rocky@cht.com.tw)
    
- For issues and reproducibility questions: please use GitHub Issues in the repository.
    
