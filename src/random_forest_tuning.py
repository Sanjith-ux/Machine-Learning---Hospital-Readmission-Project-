
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

TUNING_DIR = RESULTS_DIR / "hyperparameter_tuning"
TUNING_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD TRAINING AND TEST DATA
# ==================================================

print("Loading datasets...")

X_train = joblib.load(
    RESULTS_DIR / "X_train_processed.joblib"
)

X_test = joblib.load(
    RESULTS_DIR / "X_test_processed.joblib"
)

y_train = joblib.load(
    RESULTS_DIR / "y_train.joblib"
)

y_test = joblib.load(
    RESULTS_DIR / "y_test.joblib"
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ==================================================
# 3. LOAD PATIENT IDS FOR GROUPED CROSS-VALIDATION
# ==================================================

print("\nLoading patient group information...")

# Load patient-level split assignments
assignments = pd.read_csv(
    RESULTS_DIR / "patient_split_assignments.csv",
    usecols=["encounter_id", "split"]
)

# Load encounter IDs and patient IDs from processed data
processed_ids = pd.read_csv(
    RESULTS_DIR / "processed_data.csv",
    usecols=["encounter_id", "patient_nbr"]
)

# Recreate the same row ordering used by model preprocessing
data_with_split = processed_ids.merge(
    assignments,
    on="encounter_id",
    how="inner",
    validate="one_to_one",
    sort=False
)

# Extract training rows only
train_rows = data_with_split[
    data_with_split["split"] == "train"
].copy()

# Extract patient IDs in training row order
groups = train_rows["patient_nbr"].to_numpy()

# Verify alignment
assert len(groups) == X_train.shape[0], (
    f"Group count ({len(groups)}) does not match "
    f"training rows ({X_train.shape[0]})."
)

assert len(groups) == len(y_train), (
    "Group count does not match training target count."
)

assert pd.Series(groups).notna().all(), (
    "Missing patient IDs found in training groups."
)

print("Training patient groups:", len(groups))
print("Unique training patients:", pd.Series(groups).nunique())

print("Patient group alignment verified!")


# ==================================================
# 4. INITIALIZE RANDOM FOREST
# ==================================================

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=1
)


# ==================================================
# 5. DEFINE HYPERPARAMETER SEARCH SPACE
# ==================================================

param_grid = {
    "n_estimators": [150, 200, 300, 400],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [10, 20, 30],
    "min_samples_leaf": [5, 10, 15],
    "max_features": ["sqrt", 0.3],
    "class_weight": ["balanced", "balanced_subsample"]
}


# ==================================================
# 6. STRATIFIED GROUP CROSS-VALIDATION
# ==================================================

cv_strategy = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ==================================================
# 7. RANDOMIZED SEARCH
# ==================================================

print("\nStarting Random Forest hyperparameter tuning...")
print("Using patient-level grouped cross-validation.")
print("This may take some time.")

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=12,
    scoring="average_precision",
    cv=cv_strategy,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

# Pass patient IDs as groups
search.fit(
    X_train,
    y_train,
    groups=groups
)


# ==================================================
# 8. DISPLAY BEST PARAMETERS
# ==================================================

print("\n===== BEST RANDOM FOREST PARAMETERS =====")

print("\nBest parameters:")
print(search.best_params_)

print("\nBest grouped cross-validation PR-AUC:")
print(search.best_score_)


# Save cross-validation results
cv_results = pd.DataFrame(search.cv_results_)

cv_results.to_csv(
    TUNING_DIR / "random_forest_cv_results.csv",
    index=False
)


# ==================================================
# 9. EVALUATE BEST MODEL ON TEST DATA
# ==================================================

best_model = search.best_estimator_

y_pred = best_model.predict(X_test)

y_prob = best_model.predict_proba(X_test)[:, 1]


print("\n===== TUNED RANDOM FOREST RESULTS =====")

results = {
    "Accuracy": accuracy_score(y_test, y_pred),

    "Precision": precision_score(
        y_test,
        y_pred,
        zero_division=0
    ),

    "Recall": recall_score(
        y_test,
        y_pred,
        zero_division=0
    ),

    "F1 Score": f1_score(
        y_test,
        y_pred,
        zero_division=0
    ),

    "ROC-AUC": roc_auc_score(
        y_test,
        y_prob
    ),

    "PR-AUC": average_precision_score(
        y_test,
        y_prob
    )
}

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==================================================
# 10. SAVE TUNED MODEL AND PARAMETERS
# ==================================================

joblib.dump(
    best_model,
    MODELS_DIR / "random_forest_tuned.joblib"
)

joblib.dump(
    search.best_params_,
    TUNING_DIR / "random_forest_best_params.joblib"
)


# Save final test metrics
pd.DataFrame(
    [results]
).to_csv(
    TUNING_DIR / "random_forest_tuned_metrics.csv",
    index=False
)


print("\n==========================================")
print("RANDOM FOREST TUNING COMPLETED!")
print("==========================================")

print("\nTuned model saved successfully!")

print(f"\nResults saved in:\n{TUNING_DIR}")
