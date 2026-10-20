.PHONY: env dev install precommit test lint fmt figs clean paper paper-test nb nb-test dag

PYTHON ?= python
PIP ?= python -m pip

env:
	@echo ">>> If you use conda:"
	@echo "    conda env create -f environment.yml"
	@echo "    conda activate palpt"
	@echo ">>> Or just use pip/venv, then run: make install"

install:
	$(PIP) install -U pip
	$(PIP) install -e .[dev]
	pre-commit install

precommit:
	pre-commit run -a

test:
	pytest -q

lint:
	flake8 palatini_pt
	mypy palatini_pt

fmt:
	black .
	isort .

figs:
	@echo "Running smoke figure pipeline…"
	palpt figs --which smoke

# ---- 投稿模式（禁用所有 fallback）----
paper:
	PALPT_REQUIRE_REAL_APIS=1 PYTHONHASHSEED=0 $(PYTHON) -m scripts.make_all_figs --which all --config configs/paper_grids.yaml

paper-test:
	PALPT_REQUIRE_REAL_APIS=1 pytest -q

# --- Notebook helpers ---
NB_IPYNB := $(wildcard notebooks/*.ipynb)

nb:
	@if ! command -v jupytext >/dev/null 2>&1; then \
		echo "Error: jupytext not installed. Run 'make install' or 'pip install jupytext'."; \
		exit 1; \
	fi
	@if [ -z "$(NB_IPYNB)" ]; then \
		echo "No notebooks found under notebooks/"; exit 0; \
	fi
	jupytext --sync $(NB_IPYNB)

nb-test:
	@if ! command -v jupyter >/dev/null 2>&1; then \
		echo "Error: jupyter not installed. Run 'make install'."; \
		exit 1; \
	fi
	@if [ -z "$(NB_IPYNB)" ]; then \
		echo "No notebooks to run"; exit 0; \
	fi
	jupyter nbconvert --to notebook --execute --inplace $(NB_IPYNB)

dag:
	@if ! command -v snakemake >/dev/null 2>&1; then \
		echo "Error: snakemake not installed. Try 'pip install snakemake' or 'conda install -c conda-forge snakemake'."; \
		exit 1; \
	fi
	@if ! command -v dot >/dev/null 2>&1; then \
		echo "Error: graphviz 'dot' not found. Install via 'sudo apt-get install graphviz' or 'brew install graphviz' or 'conda install -c conda-forge graphviz'."; \
		exit 1; \
	fi
	@mkdir -p tex
	# 把 digraph 之前的提示行全部濾掉，只留下合法 DOT
	@snakemake -n --dag 2>/dev/null | sed -n '/^digraph[[:space:]]/,$$p' > tex/snakemake_dag.dot
	@if ! grep -q '^digraph' tex/snakemake_dag.dot; then \
		echo "Error: failed to capture DOT from snakemake (no 'digraph' line)."; \
		exit 1; \
	fi
	@dot -Tpdf tex/snakemake_dag.dot -o tex/snakemake_dag.pdf
	@echo "DAG written to tex/snakemake_dag.pdf"

# --- Data artifacts (App. D) ---
mixing:
	@$(PYTHON) -m scripts.mixing_matrix_extract \
		--from-json configs/coeffs/mixing_matrix_FRW.json \
		--out-csv   figs/data/mixing_matrix.csv \
		--out-meta  figs/data/mixing_matrix_meta.json \
		--lock-tol  1e-6 \
		--require-lock

data-test:
	@$(PYTHON) -m scripts.mixing_matrix_extract \
		--from-json configs/coeffs/mixing_matrix_FRW.json \
		--out-csv   figs/data/mixing_matrix.csv \
		--out-meta  figs/data/mixing_matrix_meta.json \
		--lock-tol  1e-9 \
		--require-lock \
		--no-summary

clean:
	rm -rf .mypy_cache .pytest_cache build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
