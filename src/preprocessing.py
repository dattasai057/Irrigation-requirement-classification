"""
preprocessing.py
-----------------
Shared preprocessing pipeline for the Irrigation Requirement Classification
project. Used identically by src/train.py (training) and app/streamlit_app.py
(inference), so the app never drifts from what the model was trained on.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "irrigation_required"

NUMERIC_FEATURES = [
    "soil_moisture_pct",
    "temperature_c",
    "humidity_pct",
    "rainfall_mm_last_7days",
    "days_since_last_irrigation",
]
CATEGORICAL_FEATURES = ["crop_stage", "soil_type", "crop_type"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Physically valid ranges used both to clip obvious sensor-fault outliers
# during training and to validate user input in the Streamlit app.
VALID_RANGES = {
    "soil_moisture_pct": (0.0, 60.0),
    "temperature_c": (-5.0, 50.0),
    "humidity_pct": (0.0, 100.0),
    "rainfall_mm_last_7days": (0.0, 300.0),
    "days_since_last_irrigation": (0, 60),
}

CROP_STAGE_OPTIONS = ["seedling", "vegetative", "flowering", "maturity"]
SOIL_TYPE_OPTIONS = ["sandy", "loamy", "clayey"]
CROP_TYPE_OPTIONS = ["wheat", "maize", "rice", "sugarcane", "cotton", "groundnut"]


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and apply the cleaning steps decided during EDA:
    drop exact duplicate rows, clip physically-impossible outlier values to
    the valid sensor range (rather than dropping, to preserve sample size),
    and leave missing values for the imputers inside the sklearn Pipeline
    (keeps all imputation logic in one place and prevents train/inference
    drift or leakage from statistics computed on the full dataset)."""
    df = pd.read_csv(csv_path)

    if "record_id" in df.columns:
        df = df.drop(columns=["record_id"])

    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        print(f"Dropped {removed} exact duplicate rows.")

    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            n_bad = ((df[col] < lo) | (df[col] > hi)).sum()
            if n_bad:
                print(f"Clipping {n_bad} out-of-range values in '{col}' to [{lo}, {hi}].")
            df[col] = df[col].clip(lower=lo, upper=hi)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Median/most-frequent imputation + scaling/one-hot encoding, fit only
    on training data inside a Pipeline to avoid data leakage from the test
    set into preprocessing statistics."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def split_X_y(df: pd.DataFrame):
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    return X, y
