
import pandas as pd
import re
from pathlib import Path

# =====================================================
# 1. PROJECT PATHS
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"


# =====================================================
# 2. LOAD DATA
# =====================================================

# Load AHRQ diagnosis reference
mapping = pd.read_csv(
    DATA_DIR / "$dxref 2015.csv",
    skiprows=1,
    quotechar="'",
    skipinitialspace=True,
    dtype={"ICD-9-CM CODE": str}
)

# Load CCS category labels
labels = pd.read_csv(
    DATA_DIR / "dxlabel 2015.csv"
)

# Load original hospital dataset
df = pd.read_csv(
    DATA_DIR / "diabetic_data.csv",
    low_memory=False
)


# =====================================================
# 3. DISPLAY REFERENCE INFORMATION
# =====================================================

print("\n===== AHRQ MAPPING INFORMATION =====")
print("Mapping shape:", mapping.shape)
print("Mapping columns:", mapping.columns.tolist())

print("\n===== CCS LABEL INFORMATION =====")
print("Labels shape:", labels.shape)
print(labels.head(10).to_string(index=False))


# =====================================================
# 4. NORMALIZE ICD-9-CM CODES
# =====================================================

def normalize_icd9(code):
    """
    Remove decimal points and whitespace from diagnosis codes.
    Missing and unknown codes return None.

    Note: This is exact string normalization.
    It does not reproduce all Stata icd9 clean/check rules.
    """

    if pd.isna(code):
        return None

    code = str(code).strip().upper()

    if code in ["", "?"]:
        return None

    return re.sub(r"[.\s]", "", code)


# =====================================================
# 5. PREPARE AHRQ REFERENCE LOOKUP
# =====================================================

mapping["normalized_code"] = (
    mapping["ICD-9-CM CODE"]
    .apply(normalize_icd9)
)

mapping["CCS CATEGORY"] = pd.to_numeric(
    mapping["CCS CATEGORY"],
    errors="coerce"
)

# Remove rows without a usable diagnosis code or category
reference = mapping.dropna(
    subset=["normalized_code", "CCS CATEGORY"]
).copy()

# Check whether any normalized diagnosis code has
# conflicting CCS category assignments
category_counts = (
    reference.groupby("normalized_code")["CCS CATEGORY"]
    .nunique()
)

conflicting_codes = category_counts[
    category_counts > 1
]

if not conflicting_codes.empty:
    raise ValueError(
        "Conflicting CCS categories found for normalized codes: "
        f"{conflicting_codes.index.tolist()[:20]}"
    )

# Create lookup dictionary
lookup = (
    reference.drop_duplicates("normalized_code")
    .set_index("normalized_code")["CCS CATEGORY"]
    .to_dict()
)

print("\n===== REFERENCE LOOKUP =====")
print("Unique reference diagnosis codes:", len(lookup))


# =====================================================
# 6. TEST KNOWN DIAGNOSIS CODES
# =====================================================

test_codes = [
    "250.83",
    "250.7",
    "401.9",
    "486",
    "276",
    "414",
    "428",
    "V57"
]

print("\n===== DIAGNOSIS MATCHING TEST =====")

for original in test_codes:

    cleaned = normalize_icd9(original)
    category = lookup.get(cleaned)

    print(f"\nOriginal code: {original}")
    print(f"Normalized code: {cleaned}")

    if category is not None:
        print("CCS Category:", category)

        description = reference.loc[
            reference["normalized_code"] == cleaned,
            "CCS CATEGORY DESCRIPTION"
        ].iloc[0]

        print("Description:", description)

    else:
        print("No exact match found")


# =====================================================
# 7. ANALYZE ALL THREE DIAGNOSIS COLUMNS
# =====================================================

print("\n===== DIAGNOSIS MAPPING COVERAGE =====")

diagnosis_columns = [
    "diag_1",
    "diag_2",
    "diag_3"
]

mapping_results = {}

for col in diagnosis_columns:

    # Normalize original diagnosis codes
    normalized = df[col].apply(normalize_icd9)

    # Match against AHRQ reference
    mapped = normalized.map(lookup)

    # Calculate statistics
    valid = normalized.notna()
    matched = mapped.notna() & valid
    unmatched = valid & ~matched
    missing = ~valid

    total = len(df)
    valid_count = int(valid.sum())
    matched_count = int(matched.sum())
    unmatched_count = int(unmatched.sum())
    missing_count = int(missing.sum())

    coverage = (
        matched_count / valid_count * 100
        if valid_count > 0 else 0
    )

    # Store results
    mapping_results[col] = {
        "total": total,
        "valid": valid_count,
        "matched": matched_count,
        "unmatched": unmatched_count,
        "missing": missing_count,
        "coverage": coverage
    }

    print(f"\n===== {col.upper()} =====")
    print("Total encounters:", total)
    print("Non-missing codes:", valid_count)
    print("Matched codes:", matched_count)
    print("Unmatched codes:", unmatched_count)
    print("Missing codes:", missing_count)
    print(f"Exact mapping coverage: {coverage:.2f}%")

    print("\nMost frequent unmatched codes:")

    print(
        normalized[unmatched]
        .value_counts()
        .head(15)
        .to_string()
    )


# =====================================================
# 8. FINAL SUMMARY
# =====================================================

print("\n===== FINAL MAPPING SUMMARY =====")

summary = pd.DataFrame(mapping_results).T

print(summary.to_string())

print("\nDiagnosis mapping coverage analysis completed.")

print(
    "\nIMPORTANT: These are exact normalized lookup results, "
    "not yet a verified reproduction of the complete "
    "AHRQ Stata ICD-9 cleaning procedure."
)
