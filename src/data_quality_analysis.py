
import pandas as pd

# Load dataset
df = pd.read_csv("data/diabetic_data.csv")

print("DATASET SHAPE:", df.shape)

# 1. Duplicate records
print("\n--- DUPLICATE RECORDS ---")
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate encounter IDs:", df["encounter_id"].duplicated().sum())

# 2. Missing values
print("\n--- MISSING VALUES ---")

missing = df.isin(["?"]).sum() + df.isna().sum()

missing_summary = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percentage": (missing / len(df)) * 100
})

print(missing_summary.sort_values(
    by="Missing_Percentage",
    ascending=False
).to_string())

# 3. Target distribution
print("\n--- READMISSION DISTRIBUTION ---")
print(df["readmitted"].value_counts())

# 4. Categorical values
print("\n--- CATEGORICAL COLUMN VALUES ---")

for col in ["race", "gender", "age", "readmitted",
            "diag_1", "diag_2", "diag_3"]:
    print(f"\n{col}:")
    print("Unique values:", df[col].nunique())
    print(df[col].value_counts().head(15))

# 5. Numerical statistics
print("\n--- NUMERICAL STATISTICS ---")
print(df.describe().to_string())
