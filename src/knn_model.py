
from pathlib import Path
import time
import joblib
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = RESULTS_DIR / "baseline_models"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD PREPROCESSED DATA
# ==========================================

X_train = joblib.load(RESULTS_DIR / "X_train_processed.joblib")
X_test = joblib.load(RESULTS_DIR / "X_test_processed.joblib")
y_train = joblib.load(RESULTS_DIR / "y_train.joblib")
y_test = joblib.load(RESULTS_DIR / "y_test.joblib")

print("Training KNN...")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ==========================================
# 3. BUILD KNN BASELINE
# ==========================================

model = KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",
    metric="euclidean",
    n_jobs=-1,
)

start = time.time()
model.fit(X_train, y_train)

print(f"Model fitting completed in {time.time() - start:.2f} seconds")

# ==========================================
# 4. PREDICTIONS
# ==========================================

print("Generating predictions. This may take some time...")

start = time.time()

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

print(f"Prediction time: {time.time() - start:.2f} seconds")

# ==========================================
# 5. EVALUATION
# ==========================================

metrics = {
    "Model": "KNN",
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions, zero_division=0),
    "Recall": recall_score(y_test, predictions, zero_division=0),
    "F1-Score": f1_score(y_test, predictions, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, probabilities),
    "PR-AUC": average_precision_score(y_test, probabilities),
}

print("\n===== KNN BASELINE =====")

for name, value in metrics.items():
    print(f"{name}: {value}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# ==========================================
# 6. SAVE MODEL AND METRICS
# ==========================================

joblib.dump(model, MODELS_DIR / "knn_model.joblib")

pd.DataFrame([metrics]).to_csv(
    OUTPUT_DIR / "knn_metrics.csv",
    index=False,
)

print("\nKNN model and metrics saved successfully!")
