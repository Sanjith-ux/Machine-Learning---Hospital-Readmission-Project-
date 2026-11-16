
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

OUTPUT_DIR = RESULTS_DIR / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD TEST DATA
# --------------------------------------------------

print("Loading test datasets...")

X_test = joblib.load(RESULTS_DIR / "X_test_processed.joblib")
y_test = joblib.load(RESULTS_DIR / "y_test.joblib")

# --------------------------------------------------
# 3. LOAD TRAINED MODELS
# --------------------------------------------------

models = {
    "Decision Tree": joblib.load(
        MODELS_DIR / "decision_tree_model.joblib"
    ),
    "Random Forest": joblib.load(
        MODELS_DIR / "random_forest_model.joblib"
    ),
    "XGBoost": joblib.load(
        MODELS_DIR / "xgboost_model.joblib"
    )
}

# --------------------------------------------------
# 4. EVALUATE MODELS
# --------------------------------------------------

print("\nEvaluating models...")

metrics = []
predictions = {}
probabilities = {}

for name, model in models.items():

    print(f"Evaluating {name}...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    predictions[name] = y_pred
    probabilities[name] = y_prob

    metrics.append({
        "Model": name,
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
    })

metrics_df = pd.DataFrame(metrics)

# Save evaluation results
metrics_df.to_csv(
    OUTPUT_DIR / "model_metrics.csv",
    index=False
)

print("\n===== MODEL PERFORMANCE COMPARISON =====")
print(metrics_df.round(4).to_string(index=False))

# --------------------------------------------------
# 5. PERFORMANCE COMPARISON CHART
# --------------------------------------------------

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC",
    "PR-AUC"
]

ax = metrics_df.set_index("Model")[metric_columns].plot(
    kind="bar",
    figsize=(12, 7),
    width=0.75
)

ax.set_title("Machine Learning Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_ylim(0, 1)
ax.legend(title="Evaluation Metrics")
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "performance_comparison.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 6. ROC CURVE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(9, 7))

for name in models:

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities[name]
    )

    auc = roc_auc_score(
        y_test,
        probabilities[name]
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curve_comparison.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 7. PRECISION-RECALL CURVE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(9, 7))

for name in models:

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities[name]
    )

    ap = average_precision_score(
        y_test,
        probabilities[name]
    )

    plt.plot(
        recall,
        precision,
        label=f"{name} (PR-AUC = {ap:.3f})"
    )

plt.title("Precision-Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "precision_recall_comparison.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 8. CONFUSION MATRIX COMPARISON
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 3,
    figsize=(17, 5)
)

for ax, (name, y_pred) in zip(
    axes,
    predictions.items()
):

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Not Readmitted", "Readmitted"],
        cmap="Blues",
        ax=ax,
        colorbar=False
    )

    ax.set_title(name)

plt.suptitle("Confusion Matrix Comparison")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix_comparison.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 9. FINISH
# --------------------------------------------------

print("\n======================================")
print("MODEL COMPARISON COMPLETED!")
print("======================================")

print(f"\nAll results saved in:\n{OUTPUT_DIR}")

print("\nGenerated files:")
print("1. model_metrics.csv")
print("2. performance_comparison.png")
print("3. roc_curve_comparison.png")
print("4. precision_recall_comparison.png")
print("5. confusion_matrix_comparison.png")
