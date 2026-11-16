
import os
import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
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
os.makedirs(MODELS_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD PREPROCESSED DATA
# --------------------------------------------------

print("Loading preprocessed data...")

X_train = joblib.load(
    os.path.join(RESULTS_DIR, "X_train_processed.joblib")
)
X_test = joblib.load(
    os.path.join(RESULTS_DIR, "X_test_processed.joblib")
)
y_train = joblib.load(
    os.path.join(RESULTS_DIR, "y_train.joblib")
)
y_test = joblib.load(
    os.path.join(RESULTS_DIR, "y_test.joblib")
)

# --------------------------------------------------
# 3. LOAD PATIENT GROUPS
# --------------------------------------------------

processed_data = pd.read_csv(
    os.path.join(RESULTS_DIR, "processed_data.csv")
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
# 4. VERIFY DATA ALIGNMENT
# --------------------------------------------------

print("\nChecking data alignment...")

if X_train.shape[0] != len(y_train):
    raise ValueError("X_train and y_train have different row counts.")

if X_train.shape[0] != len(groups):
    raise ValueError(
        "Patient groups do not match the training data row count."
    )

if X_test.shape[0] != len(y_test):
    raise ValueError("X_test and y_test have different row counts.")

print("Training encounters:", X_train.shape[0])
print("Testing encounters:", X_test.shape[0])
print("Unique training patients:", len(set(groups)))
print("Patient group alignment verified.")

# --------------------------------------------------
# 5. STRATIFIED GROUP CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# --------------------------------------------------
# 6. HYPERPARAMETER SEARCH SPACE
# --------------------------------------------------

param_distributions = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 15, 20, 30, None],
    "min_samples_split": [2, 10, 20, 30, 50],
    "min_samples_leaf": [1, 5, 10, 15, 20],
    "max_features": [None, "sqrt", "log2", 0.3, 0.5],
    "class_weight": [None, "balanced"]
}

# --------------------------------------------------
# 7. INITIALIZE DECISION TREE
# --------------------------------------------------

decision_tree = DecisionTreeClassifier(
    random_state=42
)

# --------------------------------------------------
# 8. RANDOMIZED SEARCH
# --------------------------------------------------

print("\nStarting Decision Tree hyperparameter tuning...")
print("This may take some time.\n")

random_search = RandomizedSearchCV(
    estimator=decision_tree,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="average_precision",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=True
)

random_search.fit(
    X_train,
    y_train,
    groups=groups
)

# --------------------------------------------------
# 9. BEST PARAMETERS
# --------------------------------------------------

best_model = random_search.best_estimator_

print("\n------------------------------------")
print("BEST DECISION TREE PARAMETERS")
print("------------------------------------")

print(random_search.best_params_)

print(
    "\nBest Grouped CV Average Precision:",
    round(random_search.best_score_, 4)
)

# --------------------------------------------------
# 10. TEST SET EVALUATION
# --------------------------------------------------

print("\nEvaluating tuned Decision Tree...")

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

print("\n------------------------------------")
print("TUNED DECISION TREE RESULTS")
print("------------------------------------")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# --------------------------------------------------
# 11. SAVE MODEL
# --------------------------------------------------

model_path = os.path.join(
    MODELS_DIR,
    "decision_tree_tuned.joblib"
)

joblib.dump(best_model, model_path)

# --------------------------------------------------
# 12. SAVE CROSS-VALIDATION RESULTS
# --------------------------------------------------

cv_results = pd.DataFrame(random_search.cv_results_)

cv_results.to_csv(
    os.path.join(TUNING_DIR, "decision_tree_cv_results.csv"),
    index=False
)

joblib.dump(
    random_search.best_params_,
    os.path.join(TUNING_DIR, "decision_tree_best_params.joblib")
)

# --------------------------------------------------
# 13. SAVE TEST METRICS
# --------------------------------------------------

metrics = pd.DataFrame([{
    "Model": "Tuned Decision Tree",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC-AUC": roc_auc,
    "PR-AUC": pr_auc,
    "Best CV Average Precision": random_search.best_score_
}])

metrics.to_csv(
    os.path.join(TUNING_DIR, "decision_tree_tuned_metrics.csv"),
    index=False
)

print("\n------------------------------------")
print("DECISION TREE TUNING COMPLETED")
print("------------------------------------")

print("Model saved:", model_path)
print("CV results saved.")
print("Best parameters saved.")
print("Test metrics saved.")

