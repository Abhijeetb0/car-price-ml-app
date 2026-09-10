from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# Load trained ML pipeline
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "model" / "car_price_model.pkl"

model = joblib.load(MODEL_PATH)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Used Car Price Prediction API",
    description="ML API for predicting used car prices",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

import os

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:5173",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Request schema
# ============================================================

class CarInput(BaseModel):

    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Used Car Price Prediction API is running"
    }


# ============================================================
# Health check
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
def predict(data: CarInput):

    input_data = pd.DataFrame(
        [
            {
                "name": data.name,
                "year": data.year,
                "km_driven": data.km_driven,
                "fuel": data.fuel,
                "seller_type": data.seller_type,
                "transmission": data.transmission,
                "owner": data.owner,
            }
        ]
    )

    prediction = model.predict(input_data)[0]

    return {
        "predicted_price": round(float(prediction), 2),
        "currency": "INR",
    }
