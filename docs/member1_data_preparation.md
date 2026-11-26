# Member 1: data acquisition, EDA and quality validation

## Dataset source and acquisition

The shared input is the UCI **Diabetes 130-US Hospitals for Years 1999–2008**
dataset (DOI 10.24432/C5230J). Encounters, rather than unique patients, are rows.
UCI describes inpatient encounters lasting 1–14 days with laboratory testing,
medication administration and a diabetes diagnosis. The downloaded CSV has 50
columns: two identifiers, the readmission outcome and 47 predictor fields.

The raw dataset and shared processed export already existed in this workspace;
Member 1 did not newly collect clinical records or regenerate the cleaned data
in this session. Obtain the raw archive from the official UCI download, extract
`diabetic_data.csv` and `IDS_mapping.csv` into ignored `data/`, and never commit
the archive or dataset. The official UCI dictionary and AHRQ 2015 reference were
retrieved in this session. AHRQ files are also stored only in ignored `data/`.
The validator records SHA-256 input/reference fingerprints for reproducibility.

Sources checked on 2026-11-26:

- [UCI dataset and variable documentation](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008)
- [Official UCI archive including IDS_mapping.csv](https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip)
- [AHRQ CCS documentation](https://hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp)
- [Official single-level 2015 CCS reference](https://hcup-us.ahrq.gov/toolssoftware/ccs/Single_Level_CCS_2015.zip)
- [pandas read_csv parsing options](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
- [Fairlearn laboratory-category descriptions](https://fairlearn.org/main/user_guide/datasets/diabetes_hospital_data.html)
  (corroborates test-not-performed semantics; its derived dataset is not our input).

## Dataset features

Identifiers are `encounter_id` and `patient_nbr`, not predictive features. Other
fields describe demographics (race, gender, age bands, weight), admission and
discharge codes, hospital stay length, payer and specialty, laboratory/procedure/
medication counts, prior utilization, three diagnosis codes, diagnosis count,
glucose and HbA1c results, individual diabetes medication changes, overall
medication change and diabetes medication use. `readmitted` is the source outcome.
`target` is the only extra column in the cleaned export, yielding 51 columns.
The shared model preprocessing excludes identifiers, the source outcome and split
labels from model features; this work does not modify that code.

## Existing cleaning decisions and justification

`src/data_preprocessing.py` uses default pandas CSV parsing, replaces `?` with
missing, converts discharge codes to numeric, excludes codes 11/19/20/21 and
creates target 1 only for `<30`. These decisions predate this contribution.
The UCI dictionary verifies the excluded dispositions:

| Code | Meaning | Original encounters excluded |
|---|---|---:|
| 11 | Expired | 1,642 |
| 19 | Expired at home; Medicaid/hospice | 8 |
| 20 | Expired in medical facility; Medicaid/hospice | 2 |
| 21 | Expired, location unknown; Medicaid/hospice | 0 |

Excluding expired dispositions defines a cohort eligible for subsequent
readmission. This is a project cohort choice, not a claim to reproduce all
exclusions in the original publication. Codes 13 and 14 describe hospice
placements and remain included under current shared behavior; any change needs
team approval. No imputation, patient splitting, target or feature behavior was
changed. The new scripts read inputs and write only aggregate reports/plots.

## EDA findings

| Quantity | Observed |
|---|---:|
| Original encounters | 101,766 |
| Excluded encounters | 1,652 |
| Cleaned encounters | 100,114 |
| Cleaned unique patients | 70,439 |
| Cleaned columns | 51 |
| Target 0 | 88,757 |
| Target 1 | 11,357 |

Original readmission categories are NO: 54,864; >30: 35,545; <30: 11,357.
The positive class is **11.34%** of cleaned encounters, an approximately
7.82:1 negative-to-positive ratio. Accuracy alone is insufficient;
the shared evaluation includes precision, recall, F1 and ranking metrics.

Source unknown tokens (empty or `?`, preserving literal `None`) affect weight
96.86%, medical specialty 49.08%, payer code 39.56% and race 2.23%.
Diagnosis missingness is 0.02%, 0.35% and 1.40% for diag_1/2/3 respectively.
The five figures under `results/eda/` show missing-value percentages, cleaned
target distribution, original and cleaned age-band distributions, cohort counts,
and original readmission categories. Age frequencies count encounters, not
unique patients. These descriptive graphs use the full cohort; they must not be
used to choose model hyperparameters against the eventual held-out test set.

## Literal None category investigation

The local pandas version is **2.3.3**. Default `read_csv` consumes
literal `None` as a missing token. In this dataset, it denotes a laboratory test
not performed, which differs from an unknown value. A read-only comparison with
`keep_default_na=False` confirmed:

| Field | Original literal None | Converted to NA by default parser | Retained-cohort None / processed blanks |
|---|---:|---:|---:|
| A1Cresult | 84,748 | 84,748 | 83,238 |
| max_glu_serum | 96,420 | 96,420 | 94,890 |

Thus the current processed export has already lost those categorical labels.
The EDA missingness graph distinguishes source unknown tokens from shared-parser
missingness rather than reporting all laboratory None values as unknown.
**No shared parsing or data file was changed.** A potential future fix is to
preserve source tokens with `keep_default_na=False` and declare `?`/empty values
explicitly missing, after team approval and downstream compatibility checks.

## Diagnosis mapping review

`src/diagnosis_mapping.py` performs exact matching after removing decimal points,
then appends one or two zeros on failure. Appending zeros is an **unverified
heuristic**: it may turn an incomplete category into a more specific diagnosis.
It must not be described as a validated clinical mapping.

`src/official_diagnosis_mapping.py` normalizes case, whitespace and decimals,
checks conflicting reference assignments and uses exact AHRQ reference lookup.
This is safer for a coverage audit, but does not prove full clinical validity or
reproduce Stata ICD-9 clean/check rules. Both existing scripts were inspected,
not modified. They do not add diagnosis features to the shared cleaned export.

The new validator independently measures aggregate lookup coverage against the
fresh official AHRQ 2015 reference on the **original cohort**, preserving code
strings. No conflicting normalized reference assignments were found.

| Column | Nonmissing | Exact matches | Exact coverage | Unmatched | Unverified zero-append candidates |
|---|---:|---:|---:|---:|---:|
| diag_1 | 101,745 | 9,495 | 9.33% | 92,250 | 88,058 |
| diag_2 | 101,408 | 13,591 | 13.40% | 87,817 | 85,308 |
| diag_3 | 100,343 | 10,744 | 10.71% | 89,599 | 88,030 |

Candidate counts are a subset of unmatched values, not accepted mappings.
The dataset includes less-specific diagnosis codes; failure to match a full
reference code does not prove an invalid diagnosis. AHRQ describes CCS as
ICD-9-CM categorization and advises its use for pre-October-2015 discharges.
Our 1999–2008 cohort is within that period, but version-specific historical
compatibility and clinical interpretation remain unverified. No categories are
invented or substituted; unmatched values stay unmatched in this audit.

## Data quality validation

`src/data_quality_validation.py` runs 18 aggregate checks, all passing on the
current files. It checks duplicate encounter IDs, missing identifiers in both
cohorts, valid source readmission categories, parseable discharge codes, excluded
dispositions, exact retained encounter membership and count, binary target,
target mapping, source-label consistency, dimensions, schema and unique-patient
count. Reports also include all-feature missingness, laboratory parsing evidence,
input fingerprints and aggregate diagnosis coverage. They never list failing
identifiers, raw rows or patient-level values. Validation exits nonzero on a
failed cohort check. Missing reference files are reported as NOT_VERIFIED, not
silently treated as successful diagnosis validation. Cohort checks passing does
not mean that the laboratory parsing issue or clinical mappings are resolved.

Reports: `results/data_quality/validation_report.json` and
`results/data_quality/validation_report.md`. Raw/processed missingness definitions
are intentionally different and explicitly recorded. Repeated patient IDs are
expected across encounters; duplicate encounter IDs are not. Snapshot dimension
and patient-count expectations must be reviewed for another dataset version.

## Actual new contributions and unchanged shared work

New Member 1 work in this session:

- Independent aggregate validation script and structured reports.
- Five aggregate EDA graphs and the separate graph-generation script.
- Source-dictionary verification, laboratory-category parsing investigation,
  exact-reference mapping coverage audit and this documentation.

The exploration, cleaning, splitting, diagnosis mapping, preprocessing and model
scripts were already shared code. Their authorship is not claimed here. None
were edited, and the existing processed dataset was not regenerated. Downloaded
reference files remain ignored. No model was trained or evaluated in this session.

## Execution commands

From the project root, with existing raw and processed inputs:

```bash
source .venv/bin/activate
python src/data_quality_validation.py
python src/member1_eda.py
```

On systems with a read-only matplotlib cache directory, prefix the EDA command
with `MPLCONFIGDIR=/tmp/member1-matplotlib`. Install dependencies using the shared
`requirements.txt` if needed. AHRQ mapping files are acquired separately into
ignored `data/`; the expected diagnosis filename is `$dxref 2015.csv`.

The already-shared preprocessing script can generate the processed export when
needed (`python src/data_preprocessing.py`), but that is not required to rerun
these new read-only checks and retains the documented None-parsing limitation.
Do not add `data/`, processed CSVs, models, serialized feature matrices or record
examples to Git. Stage only the new scripts, documentation, graphs and aggregate
reports. Patient records in older repository history are not removed by this work.
