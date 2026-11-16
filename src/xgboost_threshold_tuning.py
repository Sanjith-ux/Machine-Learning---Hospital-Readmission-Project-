
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

# ==================================================
# 1. PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

TUNING_DIR = RESULTS_DIR / "hyperparameter_tuning"
TUNING_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. LOAD DATA
# ==================================================

print("Loading datasets...")

X_train = joblib.load(RESULTS_DIR / "X_train_processed.joblib")
X_test = joblib.load(RESULTS_DIR / "X_test_processed.joblib")

y_train = np.asarray(
    joblib.load(RESULTS_DIR / "y_train.joblib")
)

y_test = np.asarray(
    joblib.load(RESULTS_DIR / "y_test.joblib")
)

best_model = joblib.load(
    MODELS_DIR / "xgboost_tuned.joblib"
)

# ==================================================
# 3. LOAD PATIENT GROUPS
# ==================================================

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
    validate="one_to_one",
    sort=False
)

groups = data_with_split.loc[
    data_with_split["split"] == "train",
    "patient_nbr"
].to_numpy()

assert len(groups) == X_train.shape[0]
assert len(groups) == len(y_train)

print("Patient groups verified:", len(groups))

# ==================================================
# 4. GENERATE OUT-OF-FOLD PREDICTIONS
# ==================================================

print("\nGenerating grouped out-of-fold predictions...")

cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

oof_probabilities = np.zeros(len(y_train))

for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X_train, y_train, groups),
    start=1
):

    print(f"Training fold {fold}/3...")

    fold_model = joblib.load(
        MODELS_DIR / "xgboost_tuned.joblib"
    )

    fold_model.fit(
        X_train[train_idx],
        y_train[train_idx]
    )

    oof_probabilities[valid_idx] = (
        fold_model.predict_proba(
            X_train[valid_idx]
        )[:, 1]
    )

# ==================================================
# 5. FIND A SUITABLE THRESHOLD
# ==================================================

print("\nSelecting threshold using training predictions...")

thresholds = np.arange(0.05, 0.51, 0.01)

threshold_results = []

for threshold in thresholds:

    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_train,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_train,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_train,
        predictions,
        zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    TUNING_DIR / "xgboost_threshold_results.csv",
    index=False
)

# Select threshold maximizing F1 score
best_row = threshold_df.loc[
    threshold_df["f1_score"].idxmax()
]

best_threshold = float(best_row["threshold"])

print("\n===== SELECTED THRESHOLD =====")
print(f"Threshold: {best_threshold:.2f}")
print(f"OOF Precision: {best_row['precision']:.4f}")
print(f"OOF Recall: {best_row['recall']:.4f}")
print(f"OOF F1 Score: {best_row['f1_score']:.4f}")

# ==================================================
# 6. FINAL TEST EVALUATION
# ==================================================

print("\nEvaluating on untouched test data...")

test_probabilities = best_model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\n===== XGBOOST THRESHOLD RESULTS =====")

print("Threshold:", best_threshold)

print(
    "Precision:",
    precision_score(y_test, test_predictions, zero_division=0)
)

print(
    "Recall:",
    recall_score(y_test, test_predictions, zero_division=0)
)

print(
    "F1 Score:",
    f1_score(y_test, test_predictions, zero_division=0)
)

print(
    "ROC-AUC:",
    roc_auc_score(y_test, test_probabilities)
)

print(
    "PR-AUC:",
    average_precision_score(y_test, test_probabilities)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

# ==================================================
# 7. SAVE RESULTS
# ==================================================

joblib.dump(
    best_threshold,
    TUNING_DIR / "xgboost_best_threshold.joblib"
)

pd.DataFrame([{
    "threshold": best_threshold,
    "precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "f1_score": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(
        y_test, test_probabilities
    ),
    "pr_auc": average_precision_score(
        y_test, test_probabilities
    )
}]).to_csv(
    TUNING_DIR / "xgboost_threshold_test_metrics.csv",
    index=False
)

print("\nThreshold optimization completed!")
print("Results saved successfully.")
