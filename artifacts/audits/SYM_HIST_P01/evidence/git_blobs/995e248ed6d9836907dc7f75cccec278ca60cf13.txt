
# palatini-pt-spurion

Reproducible codebase for **Palatini × PT-even + spurion**.  
This repo accompanies the manuscript and reproduces all C1/C2/C3 results and figures end-to-end.

**Highlights**

- ✅ **Analytic claims** (C1/C2/C3) organized as paper-checkable identities
    
- ✅ **One-command** figure regeneration (`make paper`)
    
- ✅ **Full test suite** (`41 passed`) and CI-ready
    
- ✅ **Notebooks** for demonstrations + **Colab** viewing
    
- ✅ **Snakemake DAG** of the figure pipeline (`tex/snakemake_dag.pdf`)
    

---

## TL;DR — One command

```bash
# Option A: conda (recommended)
conda env create -f environment.yml
conda activate palpt

# Option B: venv/pip
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[dev]

# Reproduce all paper figures & data
make paper

# Run tests exactly as used for the paper
make paper-test
```

Expected output (abridged):

```
Generated:
  [pdfs] figs/pdf/fig1_c1_pure_trace.pdf
  ...
  [pdfs] figs/pdf/fig9_flux_ratio.pdf
  [data] figs/data/*.csv, *.npz
41 passed in ~3–4s
```

---

## Installation

```bash
# conda
conda env create -f environment.yml
conda activate palpt

# or pip/venv
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

Check tooling:

```bash
pre-commit install && pre-commit run -a
pytest -q
palpt --help
```

---

## Reproducing the paper artifacts

|Task|Command|Output|
|---|---|---|
|Generate all figures & data (paper grids)|`make paper`|`figs/pdf/fig*.pdf`, `figs/data/*`|
|Run the exact paper test suite|`make paper-test`|`41 passed`|
|Sync notebooks (Jupytext)|`make nb`|`notebooks/*.py` updated|
|Execute notebooks headless|`make nb-test`|notebooks executed in-place|
|Build the pipeline DAG (requires `snakemake`, `graphviz`)|`make dag`|`tex/snakemake_dag.pdf`|

> 已內附 **預先生成** 的圖檔與中間數據（`figs/`），可直接編排論文。想完全重現就跑 `make paper`。

---

## Google Colab (view/run the notebooks)


- 00 — sanity checks  
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ice91/palatini-pt-spurion/blob/main/notebooks/00_sanity.ipynb)
    
- 10 — C2 symbolic demo (three-chain equivalence at quadratic order)  
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ice91/palatini-pt-spurion/blob/main/notebooks/10_c2_symbolic_demo.ipynb)
    
- 20 — Flux-ratio visualization (finite domain)  
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ice91/palatini-pt-spurion/blob/main/notebooks/20_flux_ratio.ipynb)
    
- 30 — NLO offsets and δcT2(k)\delta c_T^2(k) toy forecast  
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ice91/palatini-pt-spurion/blob/main/notebooks/30_nlo_offsets.ipynb)
    

**Colab 安裝小抄（每本筆記首格可用）**

```python
# Option 1: 安裝已公開的 GitHub 套件（最簡）
!pip -q install "git+https://github.com/ice91/palatini-pt-spurion.git"

# Option 2: clone + 本地 editable 安裝（便於開發）
!git clone https://github.com/ice91/palatini-pt-spurion.git
%cd <REPO>
!pip -q install -e .
```

> 筆記僅依賴 `numpy/sympy/matplotlib` 等常見庫，Colab 原生可用；不需要 `graphviz/snakemake`。

---

## Command-line interface

```bash
palpt --help
palpt figs --which all --config configs/paper_grids.yaml   # 生成論文圖
palpt figs --which smoke                                    # 快速 smoke
```

---

## Project layout

```
palatini-pt-spurion/
├─ palatini_pt/           # 核心邏輯（algebra / palatini / equivalence / gw / spurion / io）
├─ scripts/               # 產圖腳本（fig1–fig9 + make_all_figs）
├─ configs/               # 掃描與係數設定（含 paper_grids.yaml）
├─ figs/                  # 產出 PDF/PNG 與中間數據（含 .md5）
├─ notebooks/             # Colab 友善展示 notebook
├─ tests/                 # 單元與整合測試（41 tests）
├─ tex/                   # Snakemake DAG、TikZ 原檔
├─ Snakefile             # 圖片產製規則（fig8/fig9 例）
├─ Makefile              # 一鍵：paper / tests / nb / dag
├─ environment.yml, pyproject.toml, CITATION.cff, LICENSE
```

---

## Snakemake DAG

產製流程圖（需 `snakemake` 與 `graphviz`）：

```bash
make dag
# 輸出：tex/snakemake_dag.pdf
```

---

## Reproducibility checklist (for reviewers)

- Figures regenerate with `make paper` under the pinned environment.
    
- Tests pass with `make paper-test` (41 tests).
    
- Static artifacts provided in `figs/` with `.md5` sidecars.
    
- No shell-escape required in LaTeX; figures are external PDFs.
    
- DAG of the pipeline: `tex/snakemake_dag.pdf`.
    

---

## How to cite

See **CITATION.cff**. If you use this repository, please cite both the companion paper and this software snapshot.

---

## License

MIT (see **LICENSE**).

---

## Acknowledgments

This repository separates **paper-checkable** identities (C1/C2/C3) from **computational** artifacts. We thank the community for reproducibility best practices that inspired this layout.
