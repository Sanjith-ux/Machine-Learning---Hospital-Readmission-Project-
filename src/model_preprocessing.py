
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

# =====================================================
# 2. LOAD PROCESSED DATA AND EXISTING ASSIGNMENTS
# =====================================================

df = pd.read_csv(
    RESULTS_DIR / "processed_data.csv",
    low_memory=False
)

assignments = pd.read_csv(
    RESULTS_DIR / "patient_split_assignments.csv"
)

print("Processed dataset:", df.shape)
print("Split assignments:", assignments.shape)

# =====================================================
# 3. RESTORE ORIGINAL PATIENT-LEVEL SPLIT
# =====================================================

df = df.merge(
    assignments[["encounter_id", "split"]],
    on="encounter_id",
    how="inner",
    validate="one_to_one"
)

train = df[df["split"] == "train"].copy()
test = df[df["split"] == "test"].copy()

print("\n===== DATASET SPLIT =====")
print("Training encounters:", len(train))
print("Testing encounters:", len(test))

# Verify patient separation
overlap = set(train["patient_nbr"]).intersection(
    set(test["patient_nbr"])
)

print("Patient overlap:", len(overlap))

assert len(overlap) == 0, "Patient leakage detected!"

# =====================================================
# 4. SEPARATE FEATURES AND TARGET
# =====================================================

TARGET = "target"

EXCLUDED_COLUMNS = [
    "encounter_id",
    "patient_nbr",
    "readmitted",
    "split"
]

X_train = train.drop(
    columns=[TARGET] + EXCLUDED_COLUMNS
)

X_test = test.drop(
    columns=[TARGET] + EXCLUDED_COLUMNS
)

y_train = train[TARGET]
y_test = test[TARGET]

# =====================================================
# 5. IDENTIFY FEATURE TYPES
# =====================================================

categorical_id_columns = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id"
]

categorical_columns = (
    X_train.select_dtypes(include=["object"]).columns.tolist()
    + categorical_id_columns
)

numerical_columns = [
    col for col in X_train.select_dtypes(include=["number"]).columns
    if col not in categorical_id_columns
]

print("\nNumerical features:", numerical_columns)
print("\nCategorical features:", categorical_columns)

# =====================================================
# 6. BUILD PREPROCESSING PIPELINES
# =====================================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Missing"
    )),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=10
    ))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# =====================================================
# 7. FIT USING TRAINING DATA ONLY
# =====================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

# =====================================================
# 8. DISPLAY RESULTS
# =====================================================

print("\n===== PREPROCESSING RESULTS =====")

print("Training feature shape:", X_train_processed.shape)
print("Testing feature shape:", X_test_processed.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True) * 100)

print("\nPreprocessing completed successfully!")

# =====================================================
# 9. SAVE PREPROCESSOR AND PROCESSED FEATURES
# =====================================================

import joblib

joblib.dump(
    preprocessor,
    RESULTS_DIR / "preprocessor.joblib"
)

joblib.dump(
    X_train_processed,
    RESULTS_DIR / "X_train_processed.joblib"
)

joblib.dump(
    X_test_processed,
    RESULTS_DIR / "X_test_processed.joblib"
)

joblib.dump(
    y_train,
    RESULTS_DIR / "y_train.joblib"
)

joblib.dump(
    y_test,
    RESULTS_DIR / "y_test.joblib"
)

print("\nPreprocessor and feature datasets saved successfully!")
