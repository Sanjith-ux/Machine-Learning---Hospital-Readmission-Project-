
import pandas as pd

df = pd.read_csv(
    "results/processed_data.csv",
    low_memory=False
)

# Calculate missing-value statistics
missing = pd.DataFrame({
    "Missing_Count": df.isna().sum(),
    "Missing_Percentage": df.isna().mean() * 100
})

missing = missing.sort_values(
    by="Missing_Percentage",
    ascending=False
)

print("\n===== MISSING VALUE ANALYSIS =====")
print(missing[missing["Missing_Count"] > 0].to_string())

# Identify columns with high missing percentages
print("\n===== COLUMNS WITH MORE THAN 40% MISSING =====")

print(
    missing[missing["Missing_Percentage"] > 40]
)
