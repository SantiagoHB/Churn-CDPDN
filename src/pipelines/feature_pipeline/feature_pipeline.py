"""
SCRIPT 1: feature_pipeline.py
──────────────────────────────
Loads raw data from data/01_raw/, applies cleaning + feature engineering,
validates the schema, and saves the processed dataset to data/05_model_input/.

Usage:
    python src/pipelines/feature_pipeline/feature_pipeline.py
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "data" / "01_raw"
INTERMEDIATE_DIR = BASE_DIR / "data" / "02_intermediate"
MODEL_INPUT_DIR = BASE_DIR / "data" / "05_model_input"
MODELS_DIR = BASE_DIR / "data" / "06_models"

# ── Schema expected in the raw file ──────────────────────────────────────────
EXPECTED_COLUMNS = {
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}

# ── Feature groups ─────────────────────────────────────────────────────────────
COLS_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
COLS_BINARY_CAT = [
    "Partner",
    "Dependents",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]
COLS_NOMINAL_CAT = ["InternetService", "PaymentMethod"]
COLS_ORDINAL_CAT = ["Contract"]
COLS_NUMERIC_BINARY = ["SeniorCitizen"]
CONTRACT_ORDER = [["Month-to-month", "One year", "Two year"]]


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if any expected column is missing."""
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed. Missing columns: {missing}")
    log.info("Schema validation passed ✔")


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
    log.info(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns from {path.name}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: fix TotalCharges, drop duplicates, encode target."""
    df = df.copy()

    # Fix TotalCharges (blank strings → NaN → numeric)
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip().replace("", float("nan")),
        errors="coerce",
    )

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    before = len(df)
    df = df.drop_duplicates()
    log.info(f"Dropped {before - len(df)} duplicate rows")

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore"),
            ),
        ]
    )
    nominal_cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    )
    ordinal_cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(categories=CONTRACT_ORDER)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, COLS_NUMERIC),
            ("categoric_binary", binary_cat_pipe, COLS_BINARY_CAT),
            ("categoric_nominal", nominal_cat_pipe, COLS_NOMINAL_CAT),
            ("categoric_ordinal", ordinal_cat_pipe, COLS_ORDINAL_CAT),
            ("passthrough", "passthrough", COLS_NUMERIC_BINARY),
        ]
    )


def run() -> None:
    for d in [MODEL_INPUT_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load
    raw_path = RAW_DIR / "churn_raw_selected.csv"
    df = load_raw(raw_path)

    # 2. Validate schema
    validate_schema(df)

    # 3. Clean
    df = clean(df)

    # 4. Split features / target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {x_train.shape}  |  Test: {x_test.shape}")
    log.info(f"Churn rate — train: {y_train.mean():.2%}  | test: {y_test.mean():.2%}")

    # 5. Fit preprocessor (train only)
    preprocessor = build_preprocessor()
    preprocessor.fit(x_train)
    feature_names = list(preprocessor.get_feature_names_out())

    # 6. Transform
    x_train_t = pd.DataFrame(preprocessor.transform(x_train), columns=feature_names)
    x_test_t = pd.DataFrame(preprocessor.transform(x_test), columns=feature_names)

    # 7. Save
    x_train_t.to_parquet(MODEL_INPUT_DIR / "x_train.parquet", index=False)
    x_test_t.to_parquet(MODEL_INPUT_DIR / "x_test.parquet", index=False)
    y_train.reset_index(drop=True).to_frame().to_parquet(
        MODEL_INPUT_DIR / "y_train.parquet", index=False
    )
    y_test.reset_index(drop=True).to_frame().to_parquet(
        MODEL_INPUT_DIR / "y_test.parquet", index=False
    )

    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    with open(MODELS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    log.info("✅ Feature pipeline complete.")
    log.info(f"   Saved splits  → {MODEL_INPUT_DIR}")
    log.info(f"   Preprocessor  → {MODELS_DIR / 'preprocessor.pkl'}")


if __name__ == "__main__":
    run()
