
import pandas as pd

# Load diagnosis mapping reference
mapping = pd.read_csv(
    "data/$dxref 2015.csv",
    skiprows=1,
    quotechar="'",
    skipinitialspace=True,
    dtype={"ICD-9-CM CODE": str}
)

# Load CCS category labels
labels = pd.read_csv("data/dxlabel 2015.csv")

print("--- DIAGNOSIS MAPPING FILE ---")
print("Shape:", mapping.shape)
print("\nColumns:")
print(mapping.columns.tolist())

print("\nFirst 10 records:")
print(mapping.head(10).to_string(index=False))

print("\n--- CCS CATEGORY LABELS ---")
print("Shape:", labels.shape)
print(labels.head(10).to_string(index=False))

# Check three example diagnosis codes
print("\n--- EXAMPLE DIAGNOSIS MAPPINGS ---")

for code in ["25000", "25002", "4019", "486"]:
    result = mapping[
        mapping["ICD-9-CM CODE"].str.strip() == code
    ]

    print(f"\nICD-9-CM Code: {code}")
    print(result[
        [
            "CCS CATEGORY",
            "CCS CATEGORY DESCRIPTION",
            "ICD-9-CM CODE DESCRIPTION"
        ]
    ].to_string(index=False))

# Load original hospital dataset
df = pd.read_csv("data/diabetic_data.csv", dtype=str)

# Create a normalized mapping dictionary
mapping["ICD-9-CM CODE"] = (
    mapping["ICD-9-CM CODE"].astype(str).str.strip()
)

diagnosis_dict = dict(
    zip(mapping["ICD-9-CM CODE"], mapping["CCS CATEGORY"])
)

# Normalize diagnosis codes without changing their meaning
def normalize_code(code):
    if pd.isna(code) or str(code).strip() in ["?", ""]:
        return None

    code = str(code).strip().upper()
    return code.replace(".", "")



# Load original dataset
df = pd.read_csv("data/diabetic_data.csv", dtype=str)

# Clean mapping reference columns
mapping["ICD-9-CM CODE"] = (
    mapping["ICD-9-CM CODE"].astype(str).str.strip().str.upper()
)

mapping["CCS CATEGORY"] = (
    mapping["CCS CATEGORY"].astype(str).str.strip()
)

# Create lookup dictionary
diagnosis_dict = dict(
    zip(mapping["ICD-9-CM CODE"], mapping["CCS CATEGORY"])
)


# Normalize diagnosis codes and find mapping
def map_diagnosis(code):

    if pd.isna(code) or str(code).strip() in ["?", ""]:
        return None

    code = str(code).strip().upper().replace(".", "")

    # Try exact match first
    if code in diagnosis_dict:
        return diagnosis_dict[code]

    # Try restoring omitted trailing zeros
    for candidate in [code + "0", code + "00"]:

        if candidate in diagnosis_dict:
            return diagnosis_dict[candidate]

    # No reliable match found
    return None



# Audit exact matches and fallback matches

def audit_diagnosis(code):

    if pd.isna(code) or str(code).strip() in ["?", ""]:
        return "Missing", None, None

    original = str(code).strip().upper()
    normalized = original.replace(".", "")

    # Exact match
    if normalized in diagnosis_dict:
        return "Exact", normalized, diagnosis_dict[normalized]

    # Trailing-zero fallback
    for candidate in [normalized + "0", normalized + "00"]:
        if candidate in diagnosis_dict:
            return "Fallback", candidate, diagnosis_dict[candidate]

    return "Unmapped", None, None


for column in ["diag_1", "diag_2", "diag_3"]:

    audit = df[column].apply(audit_diagnosis)

    audit_df = pd.DataFrame(
        audit.tolist(),
        columns=["Match_Type", "Matched_Code", "CCS_Category"]
    )

    print(f"\n--- {column} AUDIT ---")
    print(audit_df["Match_Type"].value_counts())

    print("\nFallback examples:")
    print(
        pd.DataFrame({
            "Original_Code": df[column],
            "Matched_Code": audit_df["Matched_Code"],
            "CCS_Category": audit_df["CCS_Category"]
        })[audit_df["Match_Type"] == "Fallback"]
        .drop_duplicates()
        .head(20)
        .to_string(index=False)
    )
