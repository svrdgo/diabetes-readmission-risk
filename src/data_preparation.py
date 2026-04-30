"""
Data preparation pipeline for the Diabetes Readmission Risk project.
Downloads the UCI dataset, cleans it, and saves train/test splits to data/.
"""

import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = "data/"
DATASET_ID = 296
RANDOM_STATE = 42
TEST_SIZE = 0.2

SELECTED_COLUMNS = [
    "race",
    "gender",
    "age",
    "time_in_hospital",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "number_inpatient",
    "number_outpatient",
    "number_emergency",
    "diag_1",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
    "readmitted",
]

NUMERICAL_COLUMNS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "number_inpatient",
    "number_outpatient",
    "number_emergency",
]

OHE_COLUMNS = [
    "race",
    "gender",
    "diag_1_grouped",
    "A1Cresult",
    "max_glu_serum",
    "change",
    "diabetesMed",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

AGE_ORDER = [
    "[0-10)",
    "[10-20)",
    "[20-30)",
    "[30-40)",
    "[40-50)",
    "[50-60)",
    "[60-70)",
    "[70-80)",
    "[80-90)",
    "[90-100)",
]


# ---------------------------------------------------------------------------
# Step 1 — Download
# ---------------------------------------------------------------------------


def download_dataset() -> pd.DataFrame:
    dataset = fetch_ucirepo(id=DATASET_ID)

    parts = [dataset.data.features, dataset.data.targets]
    if dataset.data.ids is not None:
        parts.append(dataset.data.ids)

    df = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(f"{DATA_DIR}diabetes_raw.csv", index=False)
    print(f"  Downloaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Deduplicate patients
# ---------------------------------------------------------------------------


def deduplicate_patients(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.sort_values("encounter_id").drop_duplicates(
        subset="patient_nbr", keep="first"
    )
    df = df.drop(columns=["patient_nbr", "encounter_id"])
    print(
        f"  Deduplicated: {before:,} → {len(df):,} rows (kept first encounter per patient)"
    )
    return df


# ---------------------------------------------------------------------------
# Step 3 — Feature selection
# ---------------------------------------------------------------------------


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    # weight is >50% missing and not in SELECTED_COLUMNS — implicitly dropped here
    df = df[SELECTED_COLUMNS].copy()
    print(f"  Selected {len(SELECTED_COLUMNS) - 1} features + target")
    return df


# ---------------------------------------------------------------------------
# Step 4 — Handle missing values
# ---------------------------------------------------------------------------


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    # "?" is the missing sentinel for race and diag_1
    df["race"] = df["race"].replace("?", np.nan)
    df["diag_1"] = df["diag_1"].replace("?", np.nan)
    df = df.dropna(subset=["race", "diag_1"])

    # "None" in A1Cresult / max_glu_serum means the test was not performed —
    # it is valid data. ucimlrepo may convert the string "None" to actual NaN,
    # so restore those rows as the category string "None".
    for col in ["A1Cresult", "max_glu_serum"]:
        df[col] = df[col].fillna("None")

    print(
        f"  Dropped {before - len(df):,} rows with missing race/diag_1 → {len(df):,} rows remain"
    )
    print(f"  A1Cresult levels    : {sorted(df['A1Cresult'].unique(), key=str)}")
    print(f"  max_glu_serum levels: {sorted(df['max_glu_serum'].unique(), key=str)}")
    return df


# ---------------------------------------------------------------------------
# Step 5 — Target encoding
# ---------------------------------------------------------------------------


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df["readmitted"] = (df["readmitted"] == "<30").astype(int)
    counts = df["readmitted"].value_counts()
    pct = counts[1] / len(df) * 100
    print(
        f"  Target distribution — positive (<30 days): {counts[1]:,} ({pct:.1f}%), "
        f"negative: {counts[0]:,} ({100 - pct:.1f}%)"
    )
    return df


# ---------------------------------------------------------------------------
# Step 6 — ICD-9 grouping
# ---------------------------------------------------------------------------


def group_icd9_codes(code: str) -> str:
    code = str(code).strip()

    # Letter-prefixed codes (E-codes for external causes, V-codes for supplementary)
    if code.upper().startswith("E") or code.upper().startswith("V"):
        return "Other"

    # Diabetes: 250.xx — explicit prefix check must precede numeric range checks
    if code.startswith("250"):
        return "Diabetes"

    try:
        n = float(code)
    except ValueError:
        return "Other"

    if (390 <= n <= 459) or n == 785:
        return "Circulatory"
    if (460 <= n <= 519) or n == 786:
        return "Respiratory"
    if (520 <= n <= 579) or n == 787:
        return "Digestive"
    if 800 <= n <= 999:
        return "Injury"
    if 710 <= n <= 739:
        return "Musculoskeletal"
    if (580 <= n <= 629) or n == 788:
        return "Genitourinary"
    if 140 <= n <= 239:
        return "Neoplasms"
    return "Other"


def apply_icd9_grouping(df: pd.DataFrame) -> pd.DataFrame:
    df["diag_1_grouped"] = df["diag_1"].apply(group_icd9_codes)
    df = df.drop(columns=["diag_1"])
    print(f"  ICD-9 groups:\n{df['diag_1_grouped'].value_counts().to_string()}")
    return df


# ---------------------------------------------------------------------------
# Step 7 — Categorical encoding
# ---------------------------------------------------------------------------


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    # Age: ordinal brackets → integer rank 0–9
    age_map = {bracket: rank for rank, bracket in enumerate(AGE_ORDER)}
    df["age"] = df["age"].map(age_map)
    assert df["age"].isna().sum() == 0, "Unexpected age bracket found in data"

    # Admission ID columns are integers but represent nominal categories
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[col] = df[col].astype(str)

    # One-hot encode; keep all levels so A1Cresult_None / max_glu_serum_None are retained
    df = pd.get_dummies(df, columns=OHE_COLUMNS, drop_first=False)
    print(f"  After encoding: {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Step 8 — Split → Scale → SMOTE
# ---------------------------------------------------------------------------


def split_scale_oversample(df: pd.DataFrame):
    X = df.drop(columns=["readmitted"])
    y = df["readmitted"]

    assert X.select_dtypes(include="object").empty, (
        f"Non-numeric columns found before SMOTE: "
        f"{X.select_dtypes(include='object').columns.tolist()}"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Fit scaler on training data only to avoid leakage into test set
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERICAL_COLUMNS] = scaler.fit_transform(X_train[NUMERICAL_COLUMNS])
    X_test[NUMERICAL_COLUMNS] = scaler.transform(X_test[NUMERICAL_COLUMNS])

    print(f"  Before SMOTE — train class balance: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(
        f"  After  SMOTE — train class balance: {pd.Series(y_train_res).value_counts().to_dict()}"
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    X_train_res.to_csv(f"{DATA_DIR}X_train.csv", index=False)
    X_test.to_csv(f"{DATA_DIR}X_test.csv", index=False)
    pd.Series(y_train_res, name="readmitted").to_csv(
        f"{DATA_DIR}y_train.csv", index=False
    )
    pd.Series(y_test, name="readmitted").to_csv(f"{DATA_DIR}y_test.csv", index=False)

    print(f"  Saved: X_train {X_train_res.shape}, X_test {X_test.shape}")
    return X_train_res, X_test, y_train_res, y_test


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main():
    print("[1/7] Downloading dataset...")
    df = download_dataset()

    print("[2/7] Deduplicating patients...")
    df = deduplicate_patients(df)

    print("[3/7] Selecting features...")
    df = select_features(df)

    print("[4/7] Handling missing values...")
    df = handle_missing_values(df)

    print("[5/7] Encoding target variable...")
    df = encode_target(df)

    print("[6/7] Grouping ICD-9 codes and applying...")
    df = apply_icd9_grouping(df)

    print("[7/7] Encoding categorical features...")
    df = encode_categorical_features(df)

    print("[8/8] Scaling, splitting, and applying SMOTE...")
    split_scale_oversample(df)

    print("\nDone. Files saved to data/")


if __name__ == "__main__":
    main()
