from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# Chemin racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Chargement des artefacts ML
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "models" / "features.pkl"


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)


app = FastAPI(
    title="AI Banking Risk Assistant API",
    description="API de détection de fraude bancaire basée sur XGBoost",
    version="1.0.0",
)


class Transaction(BaseModel):
    amount: float
    time: float


@app.get("/")
def root():
    return {
        "message": "AI Banking Risk Assistant API",
        "status": "active",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(transaction: Transaction):
    # Création d'une transaction de référence
    row = pd.DataFrame(0.0, index=[0], columns=features)

    # Variables disponibles dans l'interface
    if "Amount" in row.columns:
        row["Amount"] = transaction.amount

    if "Time" in row.columns:
        row["Time"] = transaction.time

    # Prétraitement
    columns_to_scale = [
        column
        for column in ["Amount", "Time"]
        if column in row.columns
    ]

    if columns_to_scale:
        row[columns_to_scale] = scaler.transform(
            row[columns_to_scale]
        )

    # Prédiction
    prediction = int(model.predict(row)[0])
    risk_score = float(model.predict_proba(row)[0][1])

    if risk_score < 0.20:
        risk_level = "FAIBLE"
    elif risk_score < 0.50:
        risk_level = "MODÉRÉ"
    else:
        risk_level = "ÉLEVÉ"

    return {
        "prediction": "FRAUDE" if prediction == 1 else "NORMALE",
        "risk_score": round(risk_score, 4),
        "risk_score_percent": round(risk_score * 100, 2),
        "risk_level": risk_level,
    }