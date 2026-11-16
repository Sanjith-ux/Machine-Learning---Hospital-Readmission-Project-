
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TUNING_DIR = os.path.join(RESULTS_DIR, "hyperparameter_tuning")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(TUNING_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("Loading preprocessed data...")

X_train = joblib.load(
    os.path.join(RESULTS_DIR, "X_train_processed.joblib")
)
X_test = joblib.load(
    os.path.join(RESULTS_DIR, "X_test_processed.joblib")
)
y_train = np.asarray(
    joblib.load(os.path.join(RESULTS_DIR, "y_train.joblib"))
)
y_test = np.asarray(
    joblib.load(os.path.join(RESULTS_DIR, "y_test.joblib"))
)

# Load the tuned Random Forest
model_path = os.path.join(
    MODELS_DIR,
    "random_forest_tuned.joblib"
)

tuned_model = joblib.load(model_path)

# --------------------------------------------------
# 3. LOAD PATIENT GROUPS
# --------------------------------------------------

processed_data = pd.read_csv(
    os.path.join(RESULTS_DIR, "processed_data.csv"),
    low_memory=False
)

assignments = pd.read_csv(
    os.path.join(RESULTS_DIR, "patient_split_assignments.csv")
)

merged_data = processed_data.merge(
    assignments[["encounter_id", "split"]],
    on="encounter_id",
    how="inner",
    validate="one_to_one"
)

train_data = merged_data[
    merged_data["split"] == "train"
]

groups = train_data["patient_nbr"].to_numpy()

# --------------------------------------------------
# 4. VERIFY ALIGNMENT
# --------------------------------------------------

print("\nChecking data alignment...")

if X_train.shape[0] != len(y_train):
    raise ValueError("Training feature and target sizes do not match.")

if X_train.shape[0] != len(groups):
    raise ValueError("Patient groups are not aligned with X_train.")

if X_test.shape[0] != len(y_test):
    raise ValueError("Testing feature and target sizes do not match.")

print("Training encounters:", X_train.shape[0])
print("Testing encounters:", X_test.shape[0])
print("Unique training patients:", len(np.unique(groups)))
print("Patient group alignment verified.")

# --------------------------------------------------
# 5. STRATIFIED GROUP CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

oof_probabilities = np.zeros(len(y_train))

print("\nGenerating out-of-fold predictions...")

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X_train, y_train, groups),
    start=1
):
    print(f"\nTraining fold {fold}/3...")

    # Check patient separation within each fold
    fold_train_groups = set(groups[train_idx])
    fold_val_groups = set(groups[val_idx])

    if fold_train_groups.intersection(fold_val_groups):
        raise ValueError("Patient leakage detected in a CV fold.")

    # Create a fresh model using the tuned parameters
    fold_model = joblib.load(model_path)

    # Train only on this fold's training portion
    fold_model.fit(
        X_train[train_idx],
        y_train[train_idx]
    )

    # Predict probabilities for the validation portion
    oof_probabilities[val_idx] = (
        fold_model.predict_proba(X_train[val_idx])[:, 1]
    )

    print(f"Fold {fold} completed.")

print("\nOOF predictions generated successfully.")

# --------------------------------------------------
# 6. FIND BEST CLASSIFICATION THRESHOLD
# --------------------------------------------------

print("\nSearching for the best threshold...")

thresholds = np.arange(0.05, 0.501, 0.01)

threshold_results = []

for threshold in thresholds:

    oof_predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_train,
        oof_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_train,
        oof_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_train,
        oof_predictions,
        zero_division=0
    )

    threshold_results.append({
        "Threshold": round(float(threshold), 2),
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

threshold_df = pd.DataFrame(threshold_results)

best_index = threshold_df["F1 Score"].idxmax()
best_threshold = threshold_df.loc[best_index, "Threshold"]

print("\n------------------------------------")
print("OPTIMAL RANDOM FOREST THRESHOLD")
print("------------------------------------")

print(f"Best Threshold: {best_threshold:.2f}")
print(f"OOF Precision: {threshold_df.loc[best_index, 'Precision']:.4f}")
print(f"OOF Recall:    {threshold_df.loc[best_index, 'Recall']:.4f}")
print(f"OOF F1 Score:  {threshold_df.loc[best_index, 'F1 Score']:.4f}")

# --------------------------------------------------
# 7. EVALUATE ON UNTOUCHED TEST SET
# --------------------------------------------------

print("\nEvaluating on the untouched test set...")

test_probabilities = tuned_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions, zero_division=0)
recall = recall_score(y_test, test_predictions, zero_division=0)
f1 = f1_score(y_test, test_predictions, zero_division=0)
roc_auc = roc_auc_score(y_test, test_probabilities)
pr_auc = average_precision_score(y_test, test_probabilities)

cm = confusion_matrix(y_test, test_predictions)

print("\n------------------------------------")
print("RANDOM FOREST - OPTIMIZED RESULTS")
print("------------------------------------")

print(f"Selected Threshold: {best_threshold:.2f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )
)

# --------------------------------------------------
# 8. SAVE RESULTS
# --------------------------------------------------

threshold_df.to_csv(
    os.path.join(
        TUNING_DIR,
        "random_forest_threshold_results.csv"
    ),
    index=False
)

metrics = pd.DataFrame([{
    "Model": "Tuned Random Forest",
    "Threshold": best_threshold,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC-AUC": roc_auc,
    "PR-AUC": pr_auc
}])

metrics.to_csv(
    os.path.join(
        TUNING_DIR,
        "random_forest_threshold_metrics.csv"
    ),
    index=False
)

joblib.dump(
    best_threshold,
    os.path.join(
        TUNING_DIR,
        "random_forest_best_threshold.joblib"
    )
)

print("\n------------------------------------")
print("THRESHOLD OPTIMIZATION COMPLETED")
print("------------------------------------")

print("Threshold results saved.")
print("Optimized metrics saved.")
print("Best threshold saved.")
