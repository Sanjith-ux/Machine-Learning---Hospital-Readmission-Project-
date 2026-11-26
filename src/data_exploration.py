
import pandas as pd

# Load the original dataset
df = pd.read_csv("data/diabetic_data.csv")

print("\n===== DATASET DIMENSIONS =====")
print(df.shape)

print("\n===== DATASET SUMMARY =====")
print("Total records:", len(df))
print("Total features:", len(df.columns))

print("\n===== COLUMN INFORMATION =====")
df.info()

print("\n===== UNKNOWN VALUES =====")
print((df == "?").sum().sort_values(ascending=False).head(20))

print("\n===== NULL VALUES =====")
print(df.isnull().sum().sort_values(ascending=False).head(20))

print("\n===== READMISSION DISTRIBUTION =====")
print(df["readmitted"].value_counts())

print("\n===== UNIQUE PATIENTS =====")
print(df["patient_nbr"].nunique())

print("\n===== UNIQUE ENCOUNTERS =====")
print(df["encounter_id"].nunique())

