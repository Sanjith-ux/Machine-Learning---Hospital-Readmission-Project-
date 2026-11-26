# Hospital Readmission Assessment Baseline

Shared baseline for Logistic Regression, Decision Tree and KNN.

## Setup and pipeline

Install dependencies with `pip install -r requirements.txt` in a virtual environment.
Obtain the Diabetes 130-US Hospitals Dataset separately and place
`diabetic_data.csv` in `data/`. Run scripts from the project root in this order:

1. `python src/data_preprocessing.py`
2. `python src/data_splitting.py`
3. `python src/model_preprocessing.py`
4. Run the required baseline script: `src/logistic_regression_model.py`,
   `src/decision_tree_model.py`, or `src/knn_model.py`.

These commands perform processing and training when a member runs them.
No models were retrained while preparing this baseline.
Patient split assignments, processed datasets, feature matrices and trained models
are local artifacts and must not be committed. Exploration output is aggregate-only.
Existing aggregate Logistic Regression and KNN metrics were copied from the
six-model integration; no Decision Tree metric file was created or inferred.

## Contributions and provenance

This baseline copies existing implementations without changing their model logic.
Original contributions remain in the local six-model development history:

- `b454ce1`: original member-1 development commit.
- `dbfef4c`: original member-2 development commit.
- `1953615`: sanitized six-model integration snapshot used for selected files.

The baseline commit records integration work, not new authorship of these models.
The complete six-model branch is retained locally as `six-model-integration`.
`main-dev` inherits existing main history; sanitizing current files does not remove
historical patient-level output.

## Team workflow

Each member starts from the same `main-dev` baseline and works on their own branch:
`member-1`, `member-2`, `member-3`, or `member-4`.
Open pull requests with **base: main-dev** and **compare: your member branch**.
Before a pull request, fetch origin and merge `origin/main-dev` into your branch
if the shared baseline has moved. Do not commit datasets, credentials or models.
The repository owner must grant each member write access (or use fork-based PRs).
Require pull requests on main-dev through repository branch protection settings.
