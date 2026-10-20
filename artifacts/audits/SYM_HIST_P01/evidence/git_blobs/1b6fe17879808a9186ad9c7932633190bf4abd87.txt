.PHONY: env dev install precommit test lint fmt figs clean

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

clean:
	rm -rf .mypy_cache .pytest_cache build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
