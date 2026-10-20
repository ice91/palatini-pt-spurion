# palatini-pt-spurion

Reproducible codebase for **Palatini × PT-even + spurion** (C1/C2/C3).  
Goals: **可重現 / 可測試 / 可擴充 / 可讀** — 一鍵跑測試與生論文圖的腳手架。

## Quick start


# Option A: conda (recommended)
conda env create -f environment.yml
conda activate palpt

# Option B: venv/pip
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[dev]

# Check tooling:
pre-commit install
pre-commit run -a
pytest -q
palpt --help
palpt --version
palpt figs --which smoke   # smoke: no-op demo in Phase 0

Project layout (later phases)

See docs in planning (C1/C2/C3 folders to be added progressively).
In Phase 0 we only provide CLI + CI + packaging so the repo is ready to grow.

Citation

See CITATION.cff
. If you use this repository, please cite the companion paper and this software snapshot.

License

MIT — see LICENSE
.

---

## `LICENSE`（MIT）

MIT License

Copyright (c) 2025 Chien-Chih Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
