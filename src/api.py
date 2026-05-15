from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
FEATURES_PATH = PROJECT_ROOT / "models" / "best_model_features.csv"

QUALITY_MAP = {
    0: "bad",
    1: "normal",
    2: "good",
}


class PredictionRequest(BaseModel):
    features: dict[str, float | int | None] = Field(
        ...,
        description="Dictionary with recipe features. Missing model features will be filled with NaN and handled by the model pipeline.",
        examples=[
            {
                "minutes": 45,
                "n_steps": 8,
                "n_ingredients": 10,
                "calories": 350,
                "total_fat_pdv": 20,
                "sugar_pdv": 10,
                "sodium_pdv": 15,
                "protein_pdv": 12,
                "saturated_fat_pdv": 8,
                "carbs_pdv": 14,
            }
        ],
    )


class PredictionResponse(BaseModel):
    prediction_id: int
    prediction_label: str
    probabilities: dict[str, float] | None
    used_features_count: int
    missing_features_count: int
    ignored_features: list[str]


def load_feature_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")

    features_df = pd.read_csv(path)

    if "feature_name" not in features_df.columns:
        raise ValueError("best_model_features.csv must contain column 'feature_name'")

    return features_df["feature_name"].astype(str).tolist()


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    return joblib.load(path)


app = FastAPI(
    title="Recipe Quality Classification API",
    description="FastAPI service for predicting recipe quality: bad / normal / good.",
    version="1.0.0",
)


try:
    model = load_model(MODEL_PATH)
    feature_names = load_feature_names(FEATURES_PATH)
except Exception as exc:
    model = None
    feature_names = []
    startup_error = str(exc)
else:
    startup_error = None


@app.get("/")
def root():
    return {
        "service": "Recipe Quality Classification API",
        "status": "ok" if startup_error is None else "model_not_loaded",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    if startup_error is not None:
        raise HTTPException(status_code=500, detail=startup_error)

    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "features_path": str(FEATURES_PATH),
        "features_count": len(feature_names),
    }


@app.get("/model-info")
def model_info():
    if startup_error is not None:
        raise HTTPException(status_code=500, detail=startup_error)

    return {
        "model_type": type(model).__name__,
        "features_count": len(feature_names),
        "features": feature_names,
        "classes": QUALITY_MAP,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if startup_error is not None:
        raise HTTPException(status_code=500, detail=startup_error)

    incoming = request.features

    row = {}
    missing_features = []

    for feature in feature_names:
        value = incoming.get(feature, np.nan)
        if value is None:
            value = np.nan
        if feature not in incoming:
            missing_features.append(feature)
        row[feature] = value

    ignored_features = sorted([feature for feature in incoming.keys() if feature not in feature_names])

    X = pd.DataFrame([row], columns=feature_names)
    X = X.apply(pd.to_numeric, errors="coerce")

    try:
        pred_id = int(model.predict(X)[0])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    pred_label = QUALITY_MAP.get(pred_id, str(pred_id))

    probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", list(range(len(proba))))

        probabilities = {
            QUALITY_MAP.get(int(cls), str(cls)): round(float(prob), 6)
            for cls, prob in zip(classes, proba)
        }

    return PredictionResponse(
        prediction_id=pred_id,
        prediction_label=pred_label,
        probabilities=probabilities,
        used_features_count=len(feature_names),
        missing_features_count=len(missing_features),
        ignored_features=ignored_features,
    )