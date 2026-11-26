"""Read-only cohort validation. Exports aggregate counts, never record-level data."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_CODES = (11, 19, 20, 21)  # Verified in UCI IDS_mapping.csv.


def load_inputs(raw_path, processed_path):
    # Preserve source tokens for the audit, without altering shared preprocessing.
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    processed = pd.read_csv(processed_path, dtype=str, keep_default_na=False)
    return raw, processed


def missing_mask(series):
    return series.isna() | series.astype(str).str.strip().isin(
        ["", "?", "None", "NA", "NaN", "nan", "null", "NULL", "N/A"]
    )


def missing_summary(frame):
    # Literal None means test not performed in two laboratory fields, not unknown.
    missing = frame.isna() | frame.isin(["", "?"])
    return {
        col: {"count": int(missing[col].sum()),
              "percent": round(float(missing[col].mean() * 100), 4)}
        for col in frame.columns
    }


def diagnosis_audit(raw, reference_path):
    if not reference_path.exists():
        return {"status": "NOT_VERIFIED", "reason": "AHRQ reference file unavailable"}
    ref = pd.read_csv(reference_path, skiprows=1, quotechar="'",
                      skipinitialspace=True, dtype=str, keep_default_na=False)
    ref.columns = ref.columns.str.strip()

    def normalize(value):
        return re.sub(r"[.\s]", "", str(value).upper())

    codes = ref["ICD-9-CM CODE"].map(normalize)
    categories = ref["CCS CATEGORY"].str.strip()
    usable = codes.ne("") & categories.ne("")
    table = pd.DataFrame({"code": codes[usable], "category": categories[usable]})
    conflicts = int(table.groupby("code")["category"].nunique().gt(1).sum())
    lookup = set(table["code"])
    result = {"status": "EXACT_LOOKUP_AUDIT_ONLY", "conflicting_reference_codes": conflicts,
              "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
              "warning": "Trailing-zero candidates are NOT verified clinical mappings and are not applied.",
              "columns": {}}
    for col in ("diag_1", "diag_2", "diag_3"):
        normalized = raw[col].map(normalize)
        valid = ~normalized.isin(["", "?"])
        exact = valid & normalized.isin(lookup)
        fallback = valid & ~exact & (
            (normalized + "0").isin(lookup) | (normalized + "00").isin(lookup))
        result["columns"][col] = {
            "nonmissing": int(valid.sum()), "missing": int((~valid).sum()),
            "exact_matches": int(exact.sum()), "unmatched": int((valid & ~exact).sum()),
            "unverified_zero_append_candidates": int(fallback.sum()),
            "exact_coverage_percent": round(float(exact.sum() / max(int(valid.sum()), 1) * 100), 4)}
    return result


def validate(raw, processed, default_raw, reference_path):
    required = {"encounter_id", "patient_nbr", "readmitted", "discharge_disposition_id"}
    if not required.issubset(raw.columns) or not (required | {"target"}).issubset(processed.columns):
        raise ValueError("Required validation columns are missing; no records printed.")
    checks = {}

    def check(name, ok, observed):
        checks[name] = {"passed": bool(ok), "observed": observed}

    for name, frame in (("original", raw), ("cleaned", processed)):
        duplicate_count = int(frame["encounter_id"].duplicated().sum())
        check(name + "_duplicate_encounter_ids", duplicate_count == 0, duplicate_count)
        for col in ("patient_nbr", "encounter_id"):
            count = int(missing_mask(frame[col]).sum())
            check(name + "_missing_" + col, count == 0, count)
        check(name + "_readmission_categories", frame["readmitted"].isin(["NO", ">30", "<30"]).all(),
              int((~frame["readmitted"].isin(["NO", ">30", "<30"])).sum()))

    raw_discharge = pd.to_numeric(raw["discharge_disposition_id"], errors="coerce")
    clean_discharge = pd.to_numeric(processed["discharge_disposition_id"], errors="coerce")
    check("discharge_codes_parseable", raw_discharge.notna().all() and clean_discharge.notna().all(),
          int(raw_discharge.isna().sum() + clean_discharge.isna().sum()))
    retained = ~raw_discharge.isin(EXCLUDED_CODES)
    check("excluded_discharge_codes_absent", not clean_discharge.isin(EXCLUDED_CODES).any(),
          int(clean_discharge.isin(EXCLUDED_CODES).sum()))
    check("expected_cohort", set(processed["encounter_id"]) == set(raw.loc[retained, "encounter_id"])
          and len(processed) == int(retained.sum()), len(processed))
    numeric_target = pd.to_numeric(processed["target"], errors="coerce")
    check("binary_target", numeric_target.isin([0, 1]).all(),
          int((~numeric_target.isin([0, 1])).sum()))
    expected_target = processed["readmitted"].eq("<30").astype(int)
    check("target_mapping", numeric_target.eq(expected_target).all(),
          int((~numeric_target.eq(expected_target)).sum()))
    # Also check labels against the source, rather than trusting processed labels.
    source_labels = raw.drop_duplicates("encounter_id").set_index("encounter_id")["readmitted"]
    check("readmission_labels_match_source", processed["readmitted"].eq(
        processed["encounter_id"].map(source_labels)).all(),
        int((~processed["readmitted"].eq(processed["encounter_id"].map(source_labels))).sum()))
    check("original_dimensions", raw.shape == (101766, 50), list(raw.shape))
    check("cleaned_dimensions", processed.shape == (100114, 51), list(processed.shape))
    check("schema", list(processed.columns) == list(raw.columns) + ["target"], len(processed.columns))
    check("cleaned_unique_patients", processed["patient_nbr"].nunique() == 70439,
          int(processed["patient_nbr"].nunique()))
    none_audit = {}
    for col in ("A1Cresult", "max_glu_serum"):
        literal = raw[col].eq("None")
        none_audit[col] = {
            "source_literal_None": int(literal.sum()),
            "default_parser_missing": int(default_raw[col].isna().sum()),
            "literal_None_parsed_as_missing": int(default_raw.loc[literal, col].isna().sum()),
            "retained_source_literal_None": int((literal & retained).sum()),
            "processed_blank_or_missing": int((processed[col].isna() | processed[col].eq("")).sum())}
    return {
        "pandas_version": pd.__version__,
        "overall_passed": all(c["passed"] for c in checks.values()), "checks": checks,
        "counts": {"original_encounters": len(raw), "excluded_encounters": int((~retained).sum()),
                   "cleaned_encounters": len(processed), "cleaned_unique_patients": int(processed["patient_nbr"].nunique()),
                   "target_distribution": {str(k): int(v) for k, v in numeric_target.value_counts().items()},
                   "original_readmission_distribution": {str(k): int(v) for k, v in raw["readmitted"].value_counts().items()},
                   "excluded_by_code": {str(code): int(raw_discharge.eq(code).sum()) for code in EXCLUDED_CODES}},
        "missingness_definition": "Source: empty or ?; literal None preserved. Cleaned: blank/unknown includes categories lost by shared default parser.",
        "missing_values": {"original_literal_tokens": missing_summary(raw), "cleaned_export": missing_summary(processed)},
        "None_category_audit": none_audit,
        "diagnosis_mapping_audit": diagnosis_audit(raw, reference_path),
        "warnings": ["Shared default parsing loses the literal None laboratory category; no shared behavior was changed.",
                     "Dimension checks are specific to the supplied UCI cohort and must be reviewed for a different source version."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/diabetic_data.csv")
    parser.add_argument("--processed", type=Path, default=ROOT / "results/processed_data.csv")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/$dxref 2015.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results/data_quality")
    args = parser.parse_args()
    raw, processed = load_inputs(args.raw, args.processed)
    default_raw = pd.read_csv(args.raw, dtype=str)
    report = validate(raw, processed, default_raw, args.reference)
    report["input_sha256"] = {
        "raw": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        "processed": hashlib.sha256(args.processed.read_bytes()).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Aggregate data quality validation", "", "Overall: " + ("PASS" if report["overall_passed"] else "FAIL"),
             "", "Only aggregate counts are exported. No identifiers or individual records are included.",
             "", "| Check | Passed | Observed |", "|---|---|---|"]
    for name, item in report["checks"].items():
        lines.append(f"| {name} | {item['passed']} | {item['observed']} |")
    lines += ["", "## Aggregate details", "", "See validation_report.json for per-feature missingness, category parsing and diagnosis coverage.",
              "", "## Limitations", ""] + ["- " + w for w in report["warnings"]]
    (args.output / "validation_report.md").write_text("\n".join(lines) + "\n")
    print(f"Validation {'PASS' if report['overall_passed'] else 'FAIL'}: {len(report['checks'])} aggregate checks; reports saved in {args.output}")
    raise SystemExit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
