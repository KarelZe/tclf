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
- Python >= 3.10
- Install: `uv sync` (installs the package plus the `dev` dependency group)
- Dependency groups: `test`, `docs`, and `dev` (includes both)

## Testing
- Run tests: `uv run pytest`
- Run benchmarks: `uv run pytest tests/ -m benchmark --no-cov`

## Documentation
- Serve docs: `uv run zensical serve`, then access http://localhost:8000

## Development Workflow
- Lint: `uv run ruff check src tests` and `uv run ruff format --check src tests`
- Type check: `uv run ty check src`
- Format: `uv run ruff format src tests`
- Build distribution: `uv build`

For ad-hoc fixes: `uv run ruff check . --fix`. For unsafe fixes, run
`uv run ruff check . --fix --unsafe-fixes --diff` first and review the diff
before applying.

## Before Committing
- `uv run pytest` passes
- ruff and `uv run ty check src` are clean
- `uv run prek run --all-files` passes
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
- Update `docs/` and `README.md` when behaviour or API changes

## Permissions

**Allowed without asking**
- Read files, run tests, run linters and formatters
- Build the extension locally

**Ask first**
- Installing new Python packages or modifying `pyproject.toml` dependencies
- Pushing to `main`
- Deleting files
- Modifying CI workflows (`.github/workflows/`) or release configuration
