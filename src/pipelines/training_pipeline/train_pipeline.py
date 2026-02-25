"""
SCRIPT 2: train_pipeline.py
────────────────────────────
Loads processed data from data/05_model_input/, trains candidate models,
tracks experiments with MLflow, selects the best model, validates it, and
serialises artefacts to data/06_models/.

Usage:
    python src/pipelines/training_pipeline/train_pipeline.py

MLflow UI:
    mlflow ui   (then open http://localhost:5000)
"""

import json
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_DIR = BASE_DIR / "data" / "05_model_input"
MODELS_DIR = BASE_DIR / "data" / "06_models"
OUTPUT_DIR = BASE_DIR / "data" / "07_model_output"
MLFLOW_DIR = BASE_DIR / "models" / "mlruns"

RANDOM_STATE = 42
CV_FOLDS = 5
EXPERIMENT_NAME = "churn-prediction"

CANDIDATES = {
    "Logistic Regression": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
                ),
            ),
        ]
    ),
    "Gradient Boosting": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.1, max_depth=4, random_state=RANDOM_STATE
                ),
            ),
        ]
    ),
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    x_train = pd.read_parquet(INPUT_DIR / "x_train.parquet")
    x_test = pd.read_parquet(INPUT_DIR / "x_test.parquet")
    y_train = pd.read_parquet(INPUT_DIR / "y_train.parquet").squeeze()
    y_test = pd.read_parquet(INPUT_DIR / "y_test.parquet").squeeze()
    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    log.info(f"Train: {x_train.shape} | Test: {x_test.shape}")
    return x_train, x_test, y_train, y_test, feature_names


def cross_validate_model(
    model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold
) -> dict[str, float]:
    scoring = ["roc_auc", "f1", "precision", "recall"]
    scores = cross_validate(model, x_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "auc_roc": float(scores["test_roc_auc"].mean()),
        "f1": float(scores["test_f1"].mean()),
        "precision": float(scores["test_precision"].mean()),
        "recall": float(scores["test_recall"].mean()),
        "auc_std": float(scores["test_roc_auc"].std()),
    }


def model_validation(
    model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Stress-test: performance must exceed minimum thresholds."""
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    metrics = {
        "auc_roc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
    THRESHOLDS = {"auc_roc": 0.70, "f1": 0.50}
    failures = [k for k, v in THRESHOLDS.items() if metrics[k] < v]
    if failures:
        log.warning(f"Model validation FAILED for: {failures}")
    else:
        log.info("Model validation PASSED ✔")
    return metrics, y_pred, y_proba


def run() -> None:
    for d in [MODELS_DIR, OUTPUT_DIR, MLFLOW_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test, feature_names = load_data()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # ── MLflow tracking ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    cv_results = {}

    for name, model in CANDIDATES.items():
        log.info(f"Training: {name} …")
        with mlflow.start_run(run_name=name):
            # Log params
            mlflow.log_params(model.get_params(deep=False))

            # Cross-validate
            cv_scores = cross_validate_model(model, x_train, y_train, cv)
            cv_results[name] = cv_scores

            # Log CV metrics
            for metric, value in cv_scores.items():
                mlflow.log_metric(f"cv_{metric}", value)

            log.info(f"  CV AUC-ROC: {cv_scores['auc_roc']:.4f} ± {cv_scores['auc_std']:.4f}")

    # ── Select best ────────────────────────────────────────────────────────────
    best_name = max(cv_results, key=lambda n: cv_results[n]["auc_roc"])
    best_model = CANDIDATES[best_name]
    log.info(f"Best model: {best_name} (CV AUC-ROC={cv_results[best_name]['auc_roc']:.4f})")

    # ── Final train on full training set ──────────────────────────────────────
    best_model.fit(x_train, y_train)

    # ── Model validation ───────────────────────────────────────────────────────
    test_metrics, y_pred, y_proba = model_validation(best_model, x_test, y_test)
    log.info("\n" + classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # ── Log best model to MLflow ───────────────────────────────────────────────
    with mlflow.start_run(run_name=f"{best_name} [FINAL]"):
        mlflow.log_params(best_model.get_params(deep=False))
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)
        mlflow.sklearn.log_model(
            best_model, artifact_path="model", pip_requirements=["scikit-learn", "joblib"]
        )
        log.info("MLflow run logged ✔")

    # ── Save artefacts ─────────────────────────────────────────────────────────
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    metadata = {
        "model_name": best_name,
        "cv_results": {n: {k: round(v, 4) for k, v in s.items()} for n, s in cv_results.items()},
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred, "y_proba": y_proba}).to_parquet(
        OUTPUT_DIR / "test_predictions.parquet", index=False
    )

    log.info("✅ Training pipeline complete.")
    log.info(f"   Best model     → {MODELS_DIR / 'best_model.pkl'}")
    log.info(f"   Metadata       → {MODELS_DIR / 'model_metadata.json'}")
    log.info(f"   Predictions    → {OUTPUT_DIR / 'test_predictions.parquet'}")


if __name__ == "__main__":
    run()
