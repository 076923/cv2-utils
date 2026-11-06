# Contributing

Thanks for your interest in improving Python OpenCV Utilities!
By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

The package supports Python 3.11+. Any supported version works for development.

```bash
git clone https://github.com/076923/cv2-utils.git
cd cv2-utils
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[opencv,cluster]" --group dev   # requires pip >= 25.1
pre-commit install
```

With [uv](https://docs.astral.sh/uv/): `uv sync --extra opencv --extra cluster`.
Do not use `--all-extras`: it would install all four OpenCV distributions, which conflict with each other.

## Checks

```bash
pytest --cov        # tests and coverage (fails below 90%)
mypy                # type checking (strict)
ruff check .        # lint
ruff format .       # format
zensical serve      # preview the documentation at http://localhost:8000
pytest benchmarks   # timings and performance regression guards (see below)
```

### Benchmarks

`benchmarks/` is not part of the normal test run. `benchmarks/test_overhead.py` holds the regression guards: each
cv2u wrapper is timed against the plain OpenCV (or NumPy) call it wraps, in the same process, and fails when it is
much slower. These are machine-independent and run in CI. `benchmarks/test_benchmarks.py` records absolute timings
with [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) for comparing two versions on one machine:

```bash
pytest benchmarks --benchmark-autosave            # on main: save a baseline in .benchmarks/
pytest benchmarks --benchmark-compare             # on your branch: compare with the last saved run
```

CI runs the guards on every pull request and prints the timings next to those of the last run on `main` (the
`benchmark` job); the timings are informational, since runner speed varies from run to run.

CI builds the wheel once and runs the tests against it on Linux for every supported Python version,
on Windows and macOS for the newest Python, without the `cluster` extra, with the headless and contrib OpenCV
distributions, with the newest OpenCV 4, and with the minimum dependency versions (OpenCV 4.10).
It also runs mypy, checks the generated typing stub, audits the workflows with [zizmor](https://docs.zizmor.sh/)
and builds the documentation.

## Generated files

- `cv2u/_cv2_exports.pyi` re-exports OpenCV's names for type checkers. It is generated from the first OpenCV 5
  release; names it does not list (newer, or only in OpenCV 4) are typed as `Any`. CI checks that it is up to date:

  ```bash
  pip install "opencv-python==5.0.0.93"
  python tools/generate_cv2_exports.py
  ```

- `tests/data/*.png` are golden images. Regenerate them only when an output change is intended, and review the images:

  ```bash
  python tools/regenerate_goldens.py
  ```

## Pull requests

1. Create a branch from `main`.
2. Add or update tests for your change.
3. Add an entry under the unreleased version in `CHANGELOG.md`.
4. Make sure the checks above pass, then open a pull request.

## Conventions

- Public function names use camelCase to match the OpenCV API (`rotateEx`, `putTextEx`).
- Every public module defines `__all__`, and new public names are also added to `cv2u/__init__.py`.
- Add type hints and a Google-style docstring (`Args:`, `Returns:`, `Raises:`) to every public function.
  The docstrings are rendered on the documentation site.
- Validate inputs and raise `ValueError` with a clear message.

## Releasing (maintainers)

The version comes from the git tag (hatch-vcs), so there is no version number to edit.

1. Set the release date in `CHANGELOG.md` and commit.
2. Tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
3. The `Release` workflow builds the package, publishes it to PyPI with trusted publishing
   (including digital attestations) and creates a GitHub release.
