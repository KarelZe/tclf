# Agents

## Project Overview
`tclf` is a pure-Python, scikit-learn–compatible library implementing classical
trade classification rules (Lee-Ready, tick test, EMO, CLNV, quote, depth,
tradesize). The public estimator is `tclf.classical_classifier.ClassicalClassifier`.

Key constraints for any change:
- **sklearn API contract**: preserve `fit` / `predict` / `predict_proba`
  semantics, `__init__` parameter validation, and compatibility with
  `sklearn.utils.estimator_checks.check_estimator`.
- **Performance**: must scale to millions of rows. Avoid Python-level loops
  over samples; prefer vectorized NumPy / pandas operations.
- **Conventions**: input columns follow the project's naming conventions
  (see `docs/naming_conventions.md` and `README.md`).

## Environment Setup
- Python >= 3.9
- Install: `pip install -e ".[dev]"`
- Development uses `tox` with `uv` as the backend.

## Testing
- run tests in tox environment: `tox -e test`

## Documentation
- serve documentation from tox environment `tox -e docs`, then access http://localhost:8000

## Code Style
- Run tests: `tox -e test`
- Lint (ruff + mypy): `tox -e lint`
- Format: `tox -e format`
- Build distribution: `tox -e build`
- Serve docs: `tox -e doc` → http://localhost:8000

For ad-hoc fixes outside tox: `ruff check . --fix`. For unsafe fixes, run
`ruff check . --fix --unsafe-fixes --diff` first and review the diff before
applying.

## Before Committing
- `tox -e test` passes
- `tox -e lint` passes (ruff + mypy clean)
- `pre-commit run --all-files` passes
- Commit message follows Conventional Commits with a trailing emoji:
  - `feat: … ✨`  `fix: … 🐛`  `perf: … ⚡`
  - `docs: … 📝`  `test: … ✅`  `style: … 💄`  `build: … 🔧`

## Project Structure
- `src/tclf` — python source files
- `tests/` — pytest suite
- `docs/` — zensical documentation source

## PR Guidelines
- Keep diffs small and focused
- Add or update tests for changed code paths
- Update `docs/` and `README.md` when behavior or API changes

## Permissions

**Allowed without asking**
- Read files, run tests, run linters and formatters
- Build the extension locally

**Ask first**
- Installing new Python packages or modifying `pyproject.toml` dependencies
- Pushing to `main`
- Deleting files
- Modifying CI workflows (`.github/workflows/`) or release configuration
