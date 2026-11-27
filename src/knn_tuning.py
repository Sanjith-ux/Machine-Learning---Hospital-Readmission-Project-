
from pathlib import Path
import json
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 1. PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
OUTPUT = RESULTS / "hyperparameter_tuning"

MODELS.mkdir(exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD AND ALIGN TRAINING DATA
# ==========================================

data = pd.read_csv(
    RESULTS / "processed_data.csv",
    low_memory=False
)

assignments = pd.read_csv(
    RESULTS / "patient_split_assignments.csv"
)

merged = data.merge(
    assignments[["encounter_id", "patient_nbr", "split"]],
    on=["encounter_id", "patient_nbr"],
    how="inner",
    validate="one_to_one"
)

if len(merged) != len(data):
    raise ValueError("Some encounters have missing split assignments.")

train = merged.loc[merged["split"] == "train"].copy()
test = merged.loc[merged["split"] == "test"].copy()

if set(train["patient_nbr"]) & set(test["patient_nbr"]):
    raise ValueError("Patient overlap detected!")

excluded = [
    "target",
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "split",
]

X_train = train.drop(columns=excluded)
y_train = train["target"].astype(int)
groups = train["patient_nbr"]

print("Training encounters:", len(train))
print("Unique training patients:", groups.nunique())

# ==========================================
# 3. PREPROCESSING
# ==========================================

categorical_ids = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

categorical = list(dict.fromkeys(
    X_train.select_dtypes(include=["object"]).columns.tolist()
    + categorical_ids
))

numerical = [
    col for col in X_train.select_dtypes(include=["number"]).columns
    if col not in categorical
]

preprocessor = ColumnTransformer([
    (
        "numerical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        numerical,
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(
                strategy="constant",
                fill_value="Missing"
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10
            )),
        ]),
        categorical,
    ),
])

# ==========================================
# 4. KNN PIPELINE
# ==========================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsClassifier(
        metric="euclidean",
        algorithm="brute",
        n_jobs=2,
    )),
])

param_grid = {
    "model__n_neighbors": [5, 15, 31],
    "model__weights": ["uniform", "distance"],
}

cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

# ==========================================
# 5. TUNING
# ==========================================

print("\nStarting KNN tuning...")
print("Candidates:", 6)
print("Cross-validation folds:", 3)
print("Total CV fits:", 18)

start = time.time()

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="average_precision",
    cv=cv,
    n_jobs=1,
    verbose=2,
    refit=True,
    error_score="raise",
)

search.fit(X_train, y_train, groups=groups)

elapsed = time.time() - start

print("\n===== KNN TUNING RESULTS =====")
print("Time:", round(elapsed, 2), "seconds")
print("Best parameters:", search.best_params_)
print("Best grouped CV PR-AUC:", search.best_score_)

# ==========================================
# 6. SAVE RESULTS
# ==========================================

pd.DataFrame(search.cv_results_).to_csv(
    OUTPUT / "knn_cv_results.csv",
    index=False,
)

joblib.dump(
    search.best_estimator_,
    MODELS / "knn_tuned.joblib",
)

with open(
    OUTPUT / "knn_best_params.json",
    "w",
) as file:
    json.dump({
        "best_params": search.best_params_,
        "best_cv_pr_auc": float(search.best_score_),
        "tuning_seconds": elapsed,
    }, file, indent=2)

print("\nTuned KNN model and results saved successfully!")
