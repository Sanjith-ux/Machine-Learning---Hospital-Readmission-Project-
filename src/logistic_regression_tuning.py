
from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 1. PROJECT PATHS
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

data = pd.read_csv(RESULTS / "processed_data.csv", low_memory=False)
assignments = pd.read_csv(
    RESULTS / "patient_split_assignments.csv"
)

merged = data.merge(
    assignments[["encounter_id", "patient_nbr", "split"]],
    on=["encounter_id", "patient_nbr"],
    how="inner",
    validate="one_to_one",
)

if len(merged) != len(data):
    raise ValueError("Some encounters have missing split assignments.")

train = merged.loc[merged["split"] == "train"].copy()
test = merged.loc[merged["split"] == "test"].copy()

assert not set(train["patient_nbr"]) & set(test["patient_nbr"])

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
# 3. BUILD FOLD-SAFE PREPROCESSOR
# ==========================================

categorical_ids = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

categorical = list(
    dict.fromkeys(
        X_train.select_dtypes(include=["object"]).columns.tolist()
        + categorical_ids
    )
)

numerical = [
    c for c in X_train.select_dtypes(include=["number"]).columns
    if c not in categorical
]

def make_preprocessor():
    return ColumnTransformer([
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
                    fill_value="Missing",
                )),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                )),
            ]),
            categorical,
        ),
    ])

# ==========================================
# 4. MODEL PIPELINE
# ==========================================

pipeline = Pipeline([
    ("preprocessor", make_preprocessor()),
    ("model", LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        random_state=42,
    )),
])

parameters = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__class_weight": [None, "balanced"],
}

cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42,
)

# ==========================================
# 5. HYPERPARAMETER TUNING
# ==========================================

print("\nStarting Logistic Regression tuning...")
start = time.time()

search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameters,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    refit=True,
    error_score="raise",
)

with warnings.catch_warnings():
    warnings.simplefilter("always", ConvergenceWarning)
    search.fit(X_train, y_train, groups=groups)

print("\nTuning completed!")
print("Time:", round(time.time() - start, 2), "seconds")
print("Best parameters:", search.best_params_)
print("Best grouped CV PR-AUC:", search.best_score_)

# ==========================================
# 6. SAVE TUNING RESULTS
# ==========================================

pd.DataFrame(search.cv_results_).to_csv(
    OUTPUT / "logistic_regression_cv_results.csv",
    index=False,
)

joblib.dump(
    search.best_estimator_,
    MODELS / "logistic_regression_tuned.joblib",
)

with open(
    OUTPUT / "logistic_regression_best_params.json",
    "w",
) as f:
    json.dump({
        "best_params": search.best_params_,
        "best_cv_pr_auc": float(search.best_score_),
    }, f, indent=2)

print("\nTuned model and CV results saved successfully!")
