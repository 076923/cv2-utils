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
pip install -e ".[cluster]" --group dev   # requires pip >= 25.1
pre-commit install
```

With [uv](https://docs.astral.sh/uv/): `uv sync --all-extras`.

## Checks

```bash
pytest --cov        # tests and coverage (fails below 90%)
mypy                # type checking (strict)
ruff check .        # lint
ruff format .       # format
zensical serve      # preview the documentation at http://localhost:8000
```

CI builds the wheel once and runs the tests against it on Linux for every supported Python version,
on Windows and macOS for the newest Python, without the `cluster` extra, and with the minimum dependency versions.
It also runs mypy, checks the generated typing stub, audits the workflows with [zizmor](https://docs.zizmor.sh/)
and builds the documentation.

## Generated files

- `cv2u/_cv2_exports.pyi` re-exports OpenCV's names for type checkers. Regenerate it from the **minimum**
  supported OpenCV version whenever that version changes:

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
