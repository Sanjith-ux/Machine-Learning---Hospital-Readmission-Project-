
from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Load existing preprocessed data
X_train = joblib.load(RESULTS_DIR / "X_train_processed.joblib")
X_test = joblib.load(RESULTS_DIR / "X_test_processed.joblib")
y_train = joblib.load(RESULTS_DIR / "y_train.joblib")
y_test = joblib.load(RESULTS_DIR / "y_test.joblib")

print("Training Logistic Regression...")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Preserve teammate's original baseline settings
model = LogisticRegression(
    solver="liblinear",
    class_weight="balanced",
    max_iter=2000,
    random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

metrics = {
    "Model": "Logistic Regression",
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions, zero_division=0),
    "Recall": recall_score(y_test, predictions, zero_division=0),
    "F1-Score": f1_score(y_test, predictions, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, probabilities),
    "PR-AUC": average_precision_score(y_test, probabilities),
}

print("\n===== LOGISTIC REGRESSION BASELINE =====")

for name, value in metrics.items():
    print(f"{name}: {value}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# Save model and metrics
joblib.dump(model, MODELS_DIR / "logistic_regression_model.joblib")

output_dir = RESULTS_DIR / "baseline_models"
output_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame([metrics]).to_csv(
    output_dir / "logistic_regression_metrics.csv",
    index=False,
)

print("\nModel and metrics saved successfully!")
