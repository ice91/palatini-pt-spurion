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

nb:
	jupytext --sync notebooks/*.py


nb-test:
	jupyter nbconvert --to notebook --execute --inplace notebooks/00_sanity.ipynb
	jupyter nbconvert --to notebook --execute --inplace notebooks/10_c2_symbolic_demo.ipynb

dag:
	snakemake -n --dag | dot -Tpdf > tex/snakemake_dag.pdf

clean:
	rm -rf .mypy_cache .pytest_cache build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
