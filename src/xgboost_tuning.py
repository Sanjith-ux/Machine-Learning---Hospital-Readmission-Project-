
from pathlib import Path

import joblib
import pandas as pd

from xgboost import XGBClassifier
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
# 2. LOAD DATASETS
# ==================================================

print("Loading datasets...")

X_train = joblib.load(RESULTS_DIR / "X_train_processed.joblib")
X_test = joblib.load(RESULTS_DIR / "X_test_processed.joblib")

y_train = joblib.load(RESULTS_DIR / "y_train.joblib")
y_test = joblib.load(RESULTS_DIR / "y_test.joblib")

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ==================================================
# 3. LOAD PATIENT GROUPS
# ==================================================

print("\nLoading patient group information...")

assignments = pd.read_csv(
    RESULTS_DIR / "patient_split_assignments.csv",
    usecols=["encounter_id", "split"]
)

processed_ids = pd.read_csv(
    RESULTS_DIR / "processed_data.csv",
    usecols=["encounter_id", "patient_nbr"]
)

data_with_split = processed_ids.merge(
    assignments,
    on="encounter_id",
    how="inner",
    validate="one_to_one",
    sort=False
)

train_rows = data_with_split[
    data_with_split["split"] == "train"
].copy()

groups = train_rows["patient_nbr"].to_numpy()

assert len(groups) == X_train.shape[0], (
    "Patient group count does not match training rows."
)

assert len(groups) == len(y_train), (
    "Patient group count does not match training targets."
)

assert pd.Series(groups).notna().all(), (
    "Missing patient IDs found."
)

print("Training patient groups:", len(groups))
print("Unique training patients:", pd.Series(groups).nunique())

print("Patient group alignment verified!")


# ==================================================
# 4. INITIALIZE XGBOOST
# ==================================================

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=42,
    n_jobs=1,
    tree_method="hist"
)


# ==================================================
# 5. HYPERPARAMETER SEARCH SPACE
# ==================================================

param_grid = {
    "n_estimators": [150, 200, 300, 400],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2],
    "reg_lambda": [1, 2, 5]
}


# ==================================================
# 6. GROUPED CROSS-VALIDATION
# ==================================================

cv_strategy = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ==================================================
# 7. RANDOMIZED SEARCH
# ==================================================

print("\nStarting XGBoost hyperparameter tuning...")
print("Using patient-level grouped cross-validation.")
print("This may take some time.")

search = RandomizedSearchCV(
    estimator=xgb,
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

search.fit(
    X_train,
    y_train,
    groups=groups
)


# ==================================================
# 8. BEST PARAMETERS
# ==================================================

print("\n===== BEST XGBOOST PARAMETERS =====")

print("\nBest parameters:")
print(search.best_params_)

print("\nBest grouped cross-validation PR-AUC:")
print(search.best_score_)

cv_results = pd.DataFrame(search.cv_results_)

cv_results.to_csv(
    TUNING_DIR / "xgboost_cv_results.csv",
    index=False
)


# ==================================================
# 9. EVALUATE BEST MODEL
# ==================================================

best_model = search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\n===== TUNED XGBOOST RESULTS =====")

results = {
    "Accuracy": accuracy_score(y_test, y_pred),

    "Precision": precision_score(
        y_test, y_pred, zero_division=0
    ),

    "Recall": recall_score(
        y_test, y_pred, zero_division=0
    ),

    "F1 Score": f1_score(
        y_test, y_pred, zero_division=0
    ),

    "ROC-AUC": roc_auc_score(y_test, y_prob),

    "PR-AUC": average_precision_score(y_test, y_prob)
}

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==================================================
# 10. SAVE TUNED MODEL
# ==================================================

joblib.dump(
    best_model,
    MODELS_DIR / "xgboost_tuned.joblib"
)

joblib.dump(
    search.best_params_,
    TUNING_DIR / "xgboost_best_params.joblib"
)

pd.DataFrame([results]).to_csv(
    TUNING_DIR / "xgboost_tuned_metrics.csv",
    index=False
)


print("\n==========================================")
print("XGBOOST TUNING COMPLETED!")
print("==========================================")

print("\nTuned model saved successfully!")
print(f"Results saved in: {TUNING_DIR}")
