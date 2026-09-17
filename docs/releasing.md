# Release validation

Before publishing a release:

1. Update the package version in `pyproject.toml` and `nbframe/__init__.py`, the
   version assertion in `scripts/verify_distribution.py`, and the changelog.
2. Run the complete test suite with coverage and compare fixture predictions
   with the preceding revision. Investigate changed scores and withheld results.
3. Open a pull request and require every **Package checks** job to pass. The
   workflow tests Python 3.10 and 3.12 on Linux and Python 3.10 on macOS, then
   creates separate environments to install and exercise the built wheel.
4. After merging, wait for the main-branch checks. Download the `distributions`
   artifact from that exact commit. The default `python -m build` command makes
   a source archive and builds the wheel from it, checking source completeness.
5. Tag the tested commit, create a GitHub release with public change notes, and
   attach the wheel, source archive, and SHA-256 checksums. Do not reuse a version
   after it has been distributed or cited.

`python scripts/check_distribution_contents.py dist` checks package contents.
To test installation manually, create a new Conda environment with Python,
HMMER, ANARCI, and scikit-learn 1.7.2; install the wheel with pip; run `pip check`;
then, from outside this repository, run
`python /path/to/NbFrame/scripts/verify_distribution.py /path/to/NbFrame`.
The script verifies that imports resolve to the installation, exercises both
models and structure formats, and checks quality withholding and the CLI.

The scikit-learn version is pinned because the bundled estimators are serialized
Python objects. Upgrading it requires validating or regenerating those model
artifacts; version-mismatch warnings must not be suppressed. This release does
not establish support for every newer Python or dependency version.
