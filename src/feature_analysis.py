import pandas as pd

df = pd.read_csv(
    "results/processed_data.csv",
    low_memory=False
)

print("\n===== COLUMN DATA TYPES =====")
print(df.dtypes)

print("\n===== NUMERIC COLUMNS =====")
print(df.select_dtypes(include="number").columns.tolist())

print("\n===== CATEGORICAL COLUMNS =====")
print(df.select_dtypes(include=["object"]).columns.tolist())

print("\n===== MIXED TYPE CHECK =====")
for col in df.columns:
    types = df[col].dropna().map(type).nunique()
    if types > 1:
        print(col, "contains mixed types")