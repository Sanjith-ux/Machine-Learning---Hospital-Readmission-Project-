
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# 1. Load the preprocessed dataset
df = pd.read_csv(
    "results/processed_data.csv",
    low_memory=False
)

# 2. Separate patient groups
groups = df["patient_nbr"]

# 3. Create patient-level 80:20 split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, groups=groups)
)

# 4. Create training and testing datasets
train = df.iloc[train_idx].copy()
test = df.iloc[test_idx].copy()

# 5. Verify patient overlap
train_patients = set(train["patient_nbr"])
test_patients = set(test["patient_nbr"])

overlap = train_patients.intersection(test_patients)

# 6. Display split information
print("\n===== DATASET SPLIT =====")
print("Total encounters:", len(df))
print("Training encounters:", len(train))
print("Testing encounters:", len(test))

print("\nTraining patients:", train["patient_nbr"].nunique())
print("Testing patients:", test["patient_nbr"].nunique())

print("\nPatient overlap:", len(overlap))

# 7. Display target distribution
print("\n===== TRAINING TARGET DISTRIBUTION =====")
print(train["target"].value_counts(normalize=True) * 100)

print("\n===== TESTING TARGET DISTRIBUTION =====")
print(test["target"].value_counts(normalize=True) * 100)

# 8. Save shared assignments
train["split"] = "train"
test["split"] = "test"

assignments = pd.concat([train, test])

assignments[
    ["encounter_id", "patient_nbr", "split"]
].to_csv(
    "results/patient_split_assignments.csv",
    index=False
)
df = pd.read_csv(
    "results/processed_data.csv",
    low_memory=False
)