from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load model and feature names
model_path = "trained_models/Random_Forest.pkl"
features_path = "trained_models/feature_names.pkl"

try:
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
except Exception as e:
    logger.exception("Failed to load model or feature names.")
    raise RuntimeError("Model loading failed.")

categorical_cols = ['Insurance Company', 'CPT4 - Procedure', 'Diag 1', 'Diag 2', 'Modality', 'Place Of Serv']

class InputData(BaseModel):
    data: List[dict]

@app.post("/predict")
def predict(input: InputData):
    try:
        df = pd.DataFrame(input.data)

        # Validate required columns exist
        expected_cols = set(categorical_cols)
        if not expected_cols.issubset(df.columns):
            missing = expected_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        # Minimal cleaning
        df['Diag 2'] = df['Diag 2'].replace("", np.nan).fillna("None").astype(str)
        df['Modality'] = df['Modality'].replace("", np.nan).fillna("Unknown").astype(str)
        
        cols_to_drop = [col for col in ['Mod 1', 'Mod 2'] if col in df.columns]
        df = df.drop(columns=cols_to_drop)

        # One-hot encoding
        df_encoded = pd.get_dummies(df, columns=categorical_cols)

        # Add missing features
        missing_cols = [col for col in feature_names if col not in df_encoded.columns]
        if missing_cols:
            missing_df = pd.DataFrame(0, index=df_encoded.index, columns=missing_cols)
            df_encoded = pd.concat([df_encoded, missing_df], axis=1)

        df_encoded = df_encoded[feature_names]

        X = df_encoded.drop(columns=["Denied"], errors="ignore").astype(float)
        preds = model.predict(X)
        result = {"predictions": np.where(preds == 1, "Accepted", "Denied").tolist()}
        return result

    except ValueError as ve:
        logger.warning(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Unexpected error during prediction.")
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Check input format.")
