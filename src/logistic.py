import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#KNN
from sklearn.neighbors import KNeighborsClassifier

#linear SVM
from sklearn.svm import LinearSVC

# 1. Load only visits assigned to the team's training set.
data = pd.read_csv("results/processed_data.csv", low_memory=False)
split = pd.read_csv("results/patient_split_assignments.csv")
train_ids = split.loc[split["split"] == "train", ["encounter_id", "patient_nbr"]]

train = train_ids.merge(
    data,
    on=["encounter_id", "patient_nbr"],
    how="left",
    validate="one_to_one",
)
if train["target"].isna().any():
    raise ValueError("A training visit is missing its target. Check the two CSV files.")

# 2. Select the inputs and the answer to predict.
features = [
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
    "time_in_hospital",
    "num_medications",
    "number_diagnoses",
]
X = train[features]
y = train["target"]

# 3. Keep all visits from each patient on the same side.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
fit_idx, val_idx = next(splitter.split(X, y, groups=train["patient_nbr"]))

# 4. Compare both models on exactly the same validation visits.
models = {
    "Dummy baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    ),
}
# KNN
models["KNN (k=5)"] = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5),
)
#linear SVM
models["Linear SVM"] = make_pipeline(
    StandardScaler(),
    LinearSVC(
        C=1.0,
        class_weight="balanced",
        dual=False,
        max_iter=5000,
        random_state=42,
    ),
)
for name, model in models.items():
    model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
    predictions = model.predict(X.iloc[val_idx])

    print(f"\n===== {name} =====")
    print(classification_report(y.iloc[val_idx], predictions, digits=3, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y.iloc[val_idx], predictions, labels=[0, 1]))