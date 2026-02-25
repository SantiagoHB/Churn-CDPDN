"""
SCRIPT 3: inference_pipeline.py
─────────────────────────────────
FastAPI service that loads the serialised model + preprocessor and exposes a
POST /predict endpoint for real-time churn predictions.

Usage:
    uvicorn src.pipelines.inference_pipeline.inference_pipeline:app --reload

Or directly:
    python src/pipelines/inference_pipeline/inference_pipeline.py

API docs: http://localhost:8000/docs
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[3]
MODELS_DIR = BASE_DIR / "data" / "06_models"


# ── Artefact loading ───────────────────────────────────────────────────────────
def load_artefacts() -> tuple[Any, Any, list[str], dict[str, Any]]:
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    with open(MODELS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(MODELS_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    log.info(f"Loaded model: {metadata['model_name']}")
    return model, preprocessor, feature_names, metadata


model, preprocessor, feature_names, metadata = load_artefacts()


# ── Request / Response schemas ─────────────────────────────────────────────────
YesNo = Literal["Yes", "No"]
YesNoNone = Literal["Yes", "No", "No internet service"]
Lines = Literal["Yes", "No", "No phone service"]


class CustomerFeatures(BaseModel):
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: YesNo = Field(..., example="Yes")
    Dependents: YesNo = Field(..., example="No")
    tenure: int = Field(..., ge=0, le=72, example=12)
    MultipleLines: Lines = Field(..., example="No")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., example="Fiber optic")
    OnlineSecurity: YesNoNone = Field(..., example="No")
    OnlineBackup: YesNoNone = Field(..., example="No")
    DeviceProtection: YesNoNone = Field(..., example="No")
    TechSupport: YesNoNone = Field(..., example="No")
    StreamingTV: YesNoNone = Field(..., example="Yes")
    StreamingMovies: YesNoNone = Field(..., example="Yes")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., example="Month-to-month"
    )
    PaperlessBilling: YesNo = Field(..., example="Yes")
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=79.85)
    TotalCharges: float = Field(..., ge=0, example=958.20)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    label: str
    model_name: str


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Predictor",
    description="REST API for churn prediction. Wraps a scikit-learn pipeline.",
    version="1.0.0",
)


CHURN_THRESHOLD = 0.5


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": metadata["model_name"]}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "model_name": metadata["model_name"],
        "test_metrics": metadata["test_metrics"],
        "features": feature_names,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    try:
        input_df = pd.DataFrame([customer.model_dump()])
        transformed = preprocessor.transform(input_df)
        proba = float(model.predict_proba(transformed)[0, 1])
        prediction = int(proba >= CHURN_THRESHOLD)
        label = "CHURN" if prediction == 1 else "NO CHURN"
        return PredictionResponse(
            churn_probability=round(proba, 4),
            churn_prediction=prediction,
            label=label,
            model_name=metadata["model_name"],
        )
    except Exception as e:
        log.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict-batch")
def predict_batch(customers: list[CustomerFeatures]) -> list[dict[str, Any]]:
    """Batch predictions for multiple customers."""
    try:
        input_df = pd.DataFrame([c.model_dump() for c in customers])
        transformed = preprocessor.transform(input_df)
        probas = model.predict_proba(transformed)[:, 1]
        predictions = (probas >= CHURN_THRESHOLD).astype(int)
        return [
            {
                "index": i,
                "churn_probability": round(float(p), 4),
                "churn_prediction": int(pred),
                "label": "CHURN" if pred == 1 else "NO CHURN",
            }
            for i, (p, pred) in enumerate(zip(probas, predictions, strict=False))
        ]
    except Exception as e:
        log.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run("inference_pipeline:app", host="0.0.0.0", port=8000, reload=True)  # noqa: S104
