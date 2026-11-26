import pandas as pd

# Load original dataset
df = pd.read_csv("data/diabetic_data.csv")

print("Original dataset shape:", df.shape)

# 1. Convert '?' into proper missing values
df = df.replace("?", pd.NA)

# 2. Exclude expired discharge encounters
expired_codes = [11, 19, 20, 21]

# Ensure discharge codes are numeric for comparison
df["discharge_disposition_id"] = pd.to_numeric(
    df["discharge_disposition_id"],
    errors="coerce"
)

before_exclusion = len(df)

df = df[
    ~df["discharge_disposition_id"].isin(expired_codes)
].copy()

after_exclusion = len(df)

print("\n--- EXPIRED DISCHARGE EXCLUSION ---")
print("Original encounters:", before_exclusion)
print("Excluded encounters:", before_exclusion - after_exclusion)
print("Remaining encounters:", after_exclusion)

# 3. Create binary target
df["target"] = (df["readmitted"] == "<30").astype(int)

# 4. Display target distribution
print("\n--- TARGET DISTRIBUTION ---")
print(df["target"].value_counts().sort_index())

print("\nTarget percentages:")
print(df["target"].value_counts(normalize=True).sort_index() * 100)

# 5. Verify target conversion
print("\n--- ORIGINAL VS NEW TARGET ---")
print(pd.crosstab(df["readmitted"], df["target"]))

# 6. Display missing values
print("\n--- MISSING VALUES AFTER STANDARDIZATION ---")
print(df.isna().sum().sort_values(ascending=False).head(15))

# 7. Display final dataset shape
print("\n--- FINAL DATASET ---")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Unique patients:", df["patient_nbr"].nunique())
print("Unique encounters:", df["encounter_id"].nunique())
df.to_csv("results/processed_data.csv", index=False)

print("\nProcessed dataset saved successfully!")