from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI()

# Load model and feature names
model_path = "trained_models/Random_Forest.pkl"
features_path = "trained_models/feature_names.pkl"

model = joblib.load(model_path)
feature_names = joblib.load(features_path)

categorical_cols = ['Insurance Company', 'CPT4 - Procedure', 'Diag 1', 'Diag 2', 'Modality', 'Place Of Serv']

class InputData(BaseModel):
    data: List[dict]

@app.post("/predict")
def predict(input: InputData):
    df = pd.DataFrame(input.data)

    # Minimal cleaning
    df['Diag 2'] = df['Diag 2'].replace("", np.nan).fillna("None").astype(str)
    df['Modality'] = df['Modality'].replace("", np.nan).fillna("Unknown").astype(str)
    df = df.drop(columns=['Mod 1', 'Mod 2'])

    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Add missing features
    missing_cols = [col for col in feature_names if col not in df_encoded.columns]
    if missing_cols:
        missing_df = pd.DataFrame(0, index=df_encoded.index, columns=missing_cols)
        df_encoded = pd.concat([df_encoded, missing_df], axis=1)

    df_encoded = df_encoded[feature_names]

    X = df_encoded.drop(columns=["Denied"], errors="ignore").astype(float)
    preds = model.predict(X)
    return {"predictions": np.where(preds == 1, "Accepted", "Denied").tolist()}

