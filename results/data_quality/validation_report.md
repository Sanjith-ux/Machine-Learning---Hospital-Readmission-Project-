# Aggregate data quality validation

Overall: PASS

Only aggregate counts are exported. No identifiers or individual records are included.

| Check | Passed | Observed |
|---|---|---|
| original_duplicate_encounter_ids | True | 0 |
| original_missing_patient_nbr | True | 0 |
| original_missing_encounter_id | True | 0 |
| original_readmission_categories | True | 0 |
| cleaned_duplicate_encounter_ids | True | 0 |
| cleaned_missing_patient_nbr | True | 0 |
| cleaned_missing_encounter_id | True | 0 |
| cleaned_readmission_categories | True | 0 |
| discharge_codes_parseable | True | 0 |
| excluded_discharge_codes_absent | True | 0 |
| expected_cohort | True | 100114 |
| binary_target | True | 0 |
| target_mapping | True | 0 |
| readmission_labels_match_source | True | 0 |
| original_dimensions | True | [101766, 50] |
| cleaned_dimensions | True | [100114, 51] |
| schema | True | 51 |
| cleaned_unique_patients | True | 70439 |

## Aggregate details

See validation_report.json for per-feature missingness, category parsing and diagnosis coverage.

## Limitations

- Shared default parsing loses the literal None laboratory category; no shared behavior was changed.
- Dimension checks are specific to the supplied UCI cohort and must be reviewed for a different source version.
