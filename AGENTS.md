# gaze-model

Kilohertz 2-D gaze reconstruction from a scanning ophthalmoscope: a multimodal
particle filter over a frozen physics line-renderer. This is a Python research
codebase (single flat package of modules at the repo root, tests in `tests/`,
project-page generator in `docs/`). See `README.md`, `PLAN.md`, and `GOALS.md`.

## Cursor Cloud specific instructions

### Environment
- Python deps live in a virtualenv at `.venv` (created/refreshed by the startup
  update script). Activate with `source .venv/bin/activate` before running
  anything. Stack: `numpy`, `scipy`, `torch` (CPU build), `matplotlib`,
  `opencv-python-headless`, `tifffile`, `pandas`, `pytest`. `torch` is installed
  from the CPU wheel index (`https://download.pytorch.org/whl/cpu`) — there is no
  GPU in this environment.
- Modules are imported flat (e.g. `import data`, `import filter as flt`). Run
  scripts/tests from the repo root so the modules resolve.

### Big gotcha: most code requires uncommitted capture data
- The raw captures, per-person atlas, and caches (SLO `*.tiff`, `*.npz`, `*.npy`,
  the `normal/`, `calibration/`, `cache/` dirs, result videos) are several GB and
  are **gitignored — they are NOT present in a fresh clone**.
- `calib.py` runs `_CAL = calibrate()` **at import time**, and `calibrate()`
  loads the atlas (`data.load_atlas()` → looks for `SLO_*.tiff` under `normal/`).
  So **any module that imports `calib` raises `FileNotFoundError` at import**.
  This transitively blocks `filter.py`, `dynamics.py`, `losses.py`, `train.py`,
  `eval_real.py`, `eval_saccade.py`, `traj_gen.py`, `synth_stream.py`, and the
  benchmark/figure-from-real-data scripts, plus the matching tests. This is a
  missing-data condition, not an environment bug — do not "fix" it by editing
  code. To exercise those paths you must supply the private dataset.

### What runs without data
- Project page figures: `python docs/make_figures.py` (writes `docs/figures/fig_*.png`).
- Serve the project page: `python -m http.server 8000 --directory docs` then open
  `http://localhost:8000/index.html`. This is the demonstrable "application".
- Data-independent unit tests: `python -m pytest tests/test_decoder.py tests/test_noise.py`.
- `python -m pytest` over the full suite will fail at collection for the
  data-dependent files (expected without the dataset).

### Lint
- No linter config is committed; there is no project lint command.
